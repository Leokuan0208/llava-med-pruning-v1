"""Random visual token pruning: a question-agnostic baseline.

Operates in-LLM (FastV-style) at the decoder layer boundary, not at
the projector output. See day-08 writeup for why pre-LLM
(projector-output) pruning is incompatible with LLaVA-Med v1.0's
image-token injection validation.

Hook architecture
-----------------
The attention_mask is built once inside LlavaLlamaModel.forward() and
the SAME mask object is passed to every decoder layer in a loop. A
hook on layer 0 alone can shorten hidden_states (which flows from
layer to layer), but the attention_mask reaching layers 1..31 is
unchanged from the original length, causing a shape-mismatch
ValueError at layer 1.

Solution: register pre-forward hooks on ALL 32 decoder layers.
  - On the prefill pass, layer 0's hook scores+selects visual tokens,
    prunes both hidden_states and attention_mask, and stores the
    chosen `new_indices` on the method instance.
  - Layers 1..31's hooks read the stored `new_indices` and slice
    attention_mask to match. hidden_states is already shorter because
    it flows from layer 0's pruned output.
  - On non-prefill passes (single-token decode) the layer hooks no-op.

Decode-step attention_mask fix
------------------------------
HF's generate() maintains its own attention_mask outside the model,
growing it by 1 each decode step. After prefill prunes to K, this
external mask still reflects the original 256-visual-token length.
We monkey-patch prepare_inputs_for_generation to slice attention_mask
at each call so it matches our pruned KV cache.

Pruning mechanism (random)
--------------------------
For the visual block of length N (256), draw a random K-element subset
of indices [0, N), sort ascending, keep those rows. K = round(N * keep_ratio).
A dedicated torch.Generator seeded from `seed` drives the sampling.

Faithfulness notes
------------------
- Position IDs: shortening the sequence implicitly re-indexes positions
  via rotary embeddings. FastV accepts this; we do too.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .base import PruningMethod


class RandomPruning(PruningMethod):
    """Randomly retain K = round(256 * keep_ratio) visual tokens."""

    def __init__(self, keep_ratio: float = 0.5, seed: int = 1234):
        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        super().__init__(keep_ratio=keep_ratio, seed=seed)
        self.keep_ratio = keep_ratio
        self._gen = torch.Generator(device="cpu").manual_seed(seed)

        # Hook bookkeeping
        self._model_handle = None
        self._layer_handles: list = []
        self._im_start_id: Optional[int] = None

        # For the prepare_inputs_for_generation monkey-patch.
        self._pifg_owner = None
        self._orig_pifg = None

        # Per-prefill-pass state, cleared after each generate()
        self._new_indices: Optional[torch.Tensor] = None
        self._pruned_length: Optional[int] = None

        # Diagnostics
        self.n_pruned_calls = 0
        self.n_skipped_calls = 0

    @property
    def name(self) -> str:
        kr_str = f"{self.keep_ratio:.2f}".replace(".", "p")
        return f"random_kr{kr_str}"

    def attach(self, loaded: Any) -> None:
        self.n_pruned_calls = 0
        self.n_skipped_calls = 0
        self._new_indices = None
        self._pruned_length = None

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

    # ------------------------------------------------------------------
    def _patched_pifg(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        """Patched prepare_inputs_for_generation: slices attention_mask
        to match the KV cache that our pruning produced.

        Called by HF's generate() at every step (prefill + each decode).
        On prefill, _new_indices is None (set later by our layer 0 hook),
        so we pass through unchanged. On decode steps, we slice
        attention_mask to match the pruned prefill length.

        attention_mask shape at decode step t:
            input:   (B, prefill_len + t)  -- old length (e.g. 335 + t)
            wanted:  (B, pruned_len + t)   -- new length (e.g. 207 + t)
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
            # Reset per-pass state; layer 0 will populate _new_indices.
            self._new_indices = None
            self._pruned_length = None
        return None

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

        perm = torch.randperm(N, generator=self._gen)
        keep_local = perm[:K].sort().values.to(hidden_states.device)
        keep_global = keep_local + visual_lo

        L = hidden_states.shape[1]
        pre = torch.arange(0, visual_lo, device=hidden_states.device)
        post = torch.arange(visual_hi, L, device=hidden_states.device)
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
        if self._new_indices is None:
            return None
        if attention_mask is None:
            return None
        if attention_mask.ndim == 4 and attention_mask.shape[-1] == hidden_states.shape[1]:
            return None
        if attention_mask.ndim == 2 and attention_mask.shape[-1] == hidden_states.shape[1]:
            return None

        new_attention_mask = _slice_mask(attention_mask, self._new_indices)
        return _repack_layer_args(args, kwargs, hidden_states, new_attention_mask)


# ============================================================================
# Helpers shared with the qsim method.
# ============================================================================


def _extract_layer_args(args, kwargs):
    """Pull hidden_states and attention_mask from a decoder-layer call."""
    if len(args) >= 1:
        hidden_states = args[0]
    else:
        hidden_states = kwargs.get("hidden_states")
    attention_mask = kwargs.get("attention_mask", None)
    if attention_mask is None and len(args) >= 2:
        attention_mask = args[1]
    return hidden_states, attention_mask


def _slice_mask(mask, indices):
    """Slice an attention_mask along its sequence dimension(s)."""
    if mask is None:
        return None
    if mask.ndim == 4:
        m = mask.index_select(2, indices)
        m = m.index_select(3, indices)
        return m
    if mask.ndim == 2:
        return mask.index_select(1, indices)
    return mask


def _repack_layer_args(args, kwargs, new_hidden, new_mask):
    """Rebuild (args, kwargs) so the decoder layer receives the new tensors."""
    new_args = list(args)
    if len(new_args) >= 1:
        new_args[0] = new_hidden
    if len(new_args) >= 2:
        new_args[1] = new_mask
    new_kwargs = dict(kwargs)
    if "hidden_states" in new_kwargs:
        new_kwargs["hidden_states"] = new_hidden
    if "attention_mask" in new_kwargs:
        new_kwargs["attention_mask"] = new_mask
    return (tuple(new_args), new_kwargs)
