"""Question-aware visual token pruning via cosine similarity to question.

Same hook architecture as RandomPruning (in-LLM, all 32 layers
registered, layer 0 does the pruning and layers 1..31 follow with
attention_mask slicing, plus prepare_inputs_for_generation patch for
decode-step mask consistency). See random_pruning.py for the full
architecture docstring.

This file differs only in the scoring logic: visual tokens are scored
by cosine similarity with the mean-pooled question embedding.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from .base import PruningMethod
from .random_pruning import _extract_layer_args, _slice_mask, _repack_layer_args


class QuestionSimilarityPruning(PruningMethod):
    """Keep visual tokens most cosine-similar to the question embedding."""

    def __init__(self, keep_ratio: float = 0.5):
        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        super().__init__(keep_ratio=keep_ratio)
        self.keep_ratio = keep_ratio
        self._loaded = None

        self._model_handle = None
        self._layer_handles: list = []
        self._im_start_id: Optional[int] = None

        # For the prepare_inputs_for_generation monkey-patch.
        self._pifg_owner = None
        self._orig_pifg = None

        self._new_indices: Optional[torch.Tensor] = None
        self._pruned_length: Optional[int] = None

        self.n_pruned_calls = 0
        self.n_skipped_calls = 0
        self.n_fallback_calls = 0

    @property
    def name(self) -> str:
        kr_str = f"{self.keep_ratio:.2f}".replace(".", "p")
        return f"qsim_kr{kr_str}"

    def attach(self, loaded: Any) -> None:
        self.n_pruned_calls = 0
        self.n_skipped_calls = 0
        self.n_fallback_calls = 0
        self._new_indices = None
        self._pruned_length = None
        self._loaded = loaded

        tokenizer = loaded.tokenizer
        self._im_start_id = int(tokenizer.convert_tokens_to_ids(["<im_start>"])[0])

        # Monkey-patch prepare_inputs_for_generation on the top-level model
        # so we can slice attention_mask at decode steps. See _patched_pifg.
        self._pifg_owner = loaded.model
        self._orig_pifg = self._pifg_owner.prepare_inputs_for_generation
        self._pifg_owner.prepare_inputs_for_generation = self._patched_pifg

        backbone = loaded.model.model
        self._model_handle = backbone.register_forward_pre_hook(
            self._capture_input_ids_hook, with_kwargs=True,
        )
        self._layer_handles.append(
            backbone.layers[0].register_forward_pre_hook(
                self._prune_layer0_hook, with_kwargs=True,
            )
        )
        for layer in backbone.layers[1:]:
            self._layer_handles.append(
                layer.register_forward_pre_hook(
                    self._slice_mask_hook, with_kwargs=True,
                )
            )

    def detach(self, loaded: Any) -> None:
        if self._model_handle is not None:
            self._model_handle.remove()
            self._model_handle = None
        for h in self._layer_handles:
            h.remove()
        self._layer_handles = []

        # Restore the original prepare_inputs_for_generation.
        if self._pifg_owner is not None and self._orig_pifg is not None:
            self._pifg_owner.prepare_inputs_for_generation = self._orig_pifg
            self._pifg_owner = None
            self._orig_pifg = None

        self._loaded = None

    # ------------------------------------------------------------------
    def _patched_pifg(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        """Patched prepare_inputs_for_generation: slices attention_mask
        to match the KV cache that our pruning produced.

        Called by HF's generate() at every step (prefill + each decode).
        On prefill, _new_indices is None (set later by our layer 0 hook),
        so we pass through unchanged. On decode steps, we slice
        attention_mask to match the pruned prefill length.
        """
        # Clear stale pruning state at the start of a new generation.
        # past_key_values is None on prefill (first call of a new sample);
        # any _new_indices left over from a previous sample is invalid for
        # this one and would cause the wrong attention_mask slice.
        if past_key_values is None:
            self._new_indices = None
            self._pruned_length = None

        out = self._orig_pifg(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            **kwargs,
        )

        # Only post-process if we have a pruning record from prefill.
        if self._new_indices is None or self._pruned_length is None:
            return out

        am = out.get("attention_mask", None)
        if am is None or am.ndim != 2:
            return out

        old_prefill_len = int(self.current_input_ids.shape[1])
        cur_total_len = am.shape[1]
        decode_t = cur_total_len - old_prefill_len

        expected_pruned_total = self._pruned_length + decode_t
        if cur_total_len == expected_pruned_total:
            return out

        prefill_part = am[:, :old_prefill_len].index_select(1, self._new_indices)
        decode_part = am[:, old_prefill_len:]
        new_am = torch.cat([prefill_part, decode_part], dim=1)
        out["attention_mask"] = new_am
        return out

    # ------------------------------------------------------------------
    def _capture_input_ids_hook(self, module, args, kwargs):
        ids = kwargs.get("input_ids", None)
        if ids is None and len(args) > 0:
            ids = args[0]
        if isinstance(ids, torch.Tensor) and ids.ndim == 2 and ids.shape[1] > 1:
            self.current_input_ids = ids
            self._new_indices = None
            self._pruned_length = None
        return None

    def _question_token_ids(self, question: str) -> torch.Tensor:
        tokenizer = self._loaded.tokenizer
        return tokenizer(question, add_special_tokens=False, return_tensors="pt").input_ids[0]

    # ------------------------------------------------------------------
    def _prune_layer0_hook(self, module, args, kwargs):
        hidden_states, attention_mask = _extract_layer_args(args, kwargs)

        if hidden_states is None or hidden_states.shape[1] <= 1:
            self.n_skipped_calls += 1
            return None
        if self.current_input_ids is None or self.keep_ratio >= 1.0:
            self.n_skipped_calls += 1
            return None

        input_ids = self.current_input_ids[0]
        start_positions = (input_ids == self._im_start_id).nonzero(as_tuple=True)[0]
        if start_positions.numel() == 0:
            self.n_skipped_calls += 1
            return None

        start_pos = int(start_positions[0].item())
        N = 256
        visual_lo = start_pos + 1
        visual_hi = start_pos + 1 + N
        K = max(1, round(N * self.keep_ratio))

        device = hidden_states.device
        dtype = hidden_states.dtype
        v_block = hidden_states[0, visual_lo:visual_hi, :]

        if self.current_question is None:
            self.n_fallback_calls += 1
            keep_local = torch.arange(K, device=device)
        else:
            q_ids = self._question_token_ids(self.current_question).to(device)
            if q_ids.numel() == 0:
                self.n_fallback_calls += 1
                keep_local = torch.arange(K, device=device)
            else:
                embed_layer = self._loaded.model.get_input_embeddings()
                with torch.no_grad():
                    q_embeds = embed_layer(q_ids).to(dtype)
                    q_vec = q_embeds.mean(dim=0)
                    q_norm = F.normalize(q_vec, dim=0)
                    v_norm = F.normalize(v_block, dim=-1)
                    scores = v_norm @ q_norm
                topk = scores.topk(k=K).indices
                keep_local, _ = topk.sort()

        keep_global = keep_local + visual_lo
        L = hidden_states.shape[1]
        pre = torch.arange(0, visual_lo, device=device)
        post = torch.arange(visual_hi, L, device=device)
        new_indices = torch.cat([pre, keep_global, post])

        self._new_indices = new_indices
        self._pruned_length = int(new_indices.numel())

        new_hidden = hidden_states.index_select(1, new_indices)
        new_attention_mask = _slice_mask(attention_mask, new_indices)

        self.n_pruned_calls += 1
        return _repack_layer_args(args, kwargs, new_hidden, new_attention_mask)

    # ------------------------------------------------------------------
    def _slice_mask_hook(self, module, args, kwargs):
        hidden_states, attention_mask = _extract_layer_args(args, kwargs)
        if hidden_states is None or hidden_states.shape[1] <= 1:
            return None
        if self._new_indices is None or attention_mask is None:
            return None
        if attention_mask.ndim == 4 and attention_mask.shape[-1] == hidden_states.shape[1]:
            return None
        if attention_mask.ndim == 2 and attention_mask.shape[-1] == hidden_states.shape[1]:
            return None
        new_attention_mask = _slice_mask(attention_mask, self._new_indices)
        return _repack_layer_args(args, kwargs, hidden_states, new_attention_mask)
