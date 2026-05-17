"""Per-sample inference runner for LLaVA-Med v1.0.

Port of LLaVA-Med-v1.0/llava/eval/model_vqa_med.py, adapted to the
harness's `LoadedModel` + `VQASample` abstractions.

Faithful to v1.0's reference inference path in every detail that
affects scoring:
  - Prompt: `<question>\\n<im_start><im_patch>*256<im_end>` (or bare
    `<im_patch>*256` if mm_use_im_start_end=False), wrapped in the
    `simple` conv template.
  - Decode: do_sample=True, temperature=0.7, max_new_tokens=1024,
    KeywordsStoppingCriteria(['###']).
  - Post-process: slice off input tokens, decode with skip_special_tokens,
    truncate at first `conv.sep`.

Notes on faithfulness:
  - Stochastic decoding (T=0.7) means run-to-run scores will vary by
    1-2 points. We seed at the start for within-project reproducibility,
    but the paper's headline 0.84 was a single run; we accept that
    variance for paper comparison.
  - `model_vqa_med.py`'s optional `--answer-prompter` two-pass mode
    is NOT used for the paper's headline numbers. We skip it here.
  - Image preprocessing uses HuggingFace CLIPImageProcessor (loaded
    in model_loader.py). v1.0 does NOT use open_clip at inference time,
    despite open_clip being imported at module-load.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image
from transformers import StoppingCriteria

from llava.conversation import conv_templates

from eval.model_loader import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    LoadedModel,
)


# =============================================================================
# Stopping criterion: ported byte-for-byte from model_vqa_med.py
# =============================================================================


class KeywordsStoppingCriteria(StoppingCriteria):
    """Stop generation when any keyword appears in the decoded suffix.

    The criterion lazily captures `start_len` on first call (input tokens),
    then on each subsequent step decodes the new tokens (output_ids beyond
    start_len) and checks for any of the keyword strings.

    Behaviour subtlety inherited from v1.0: the first call returns False
    unconditionally because `start_len` is set but `outputs` isn't computed
    until the next call. So the very first newly-generated token isn't
    checked. This is faithful to the reference; we keep it.
    """

    def __init__(self, keywords: List[str], tokenizer, input_ids: torch.Tensor):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len: Optional[int] = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor,
                 **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(
                output_ids[:, self.start_len:], skip_special_tokens=True
            )[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


# =============================================================================
# Result types
# =============================================================================


@dataclass
class Prediction:
    """One prediction record, shaped to feed score_predictions(...) directly."""
    question_id: str
    text: str               # the model's decoded answer (post-truncation)
    prompt: str             # original question (sans image tokens), for logging
    latency_ms: float       # wall-clock generation time
    n_input_tokens: int     # for diagnostics
    n_output_tokens: int    # for diagnostics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "text": self.text,
            "prompt": self.prompt,
            "latency_ms": self.latency_ms,
            "n_input_tokens": self.n_input_tokens,
            "n_output_tokens": self.n_output_tokens,
        }


# =============================================================================
# The Runner
# =============================================================================


class Runner:
    """Per-sample inference loop for LLaVA-Med v1.0."""

    def __init__(self, loaded: LoadedModel, seed: int = 42):
        """Initialize from a loaded model bundle.

        Args:
            loaded: result of model_loader.load_llava_med_v1(...).
            seed: torch RNG seed for stochastic-decode reproducibility
                  within this project. The paper's number was a single
                  unseeded run, so this only matters for our own A/B
                  comparisons between pruning configs.
        """
        self.loaded = loaded
        self.model = loaded.model
        self.tokenizer = loaded.tokenizer
        self.image_processor = loaded.image_processor
        self.conv_mode = loaded.conv_mode
        self.image_token_len = loaded.image_token_len
        self.mm_use_im_start_end = loaded.mm_use_im_start_end

        # Cache the conv template lookup. Mutating loaded.conv_mode mid-run
        # would invalidate this, but no caller does that today.
        self.conv_template = conv_templates[self.conv_mode]

        # Seed torch RNG so stochastic decoding is reproducible within a run.
        # Use a generator-style seed: the global seed is set once, and every
        # generate call samples from the (now deterministic) global RNG.
        # Per-sample re-seeding is NOT what v1.0 does and would change scores.
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # -------------------------------------------------------------------------
    # Prompt construction
    # -------------------------------------------------------------------------

    def _build_prompt(self, question: str) -> str:
        """Build the full v1.0 prompt for a single (image, question) pair.

        The prompt sequence is:
          question_text + '\\n' + image_patch_block
        — then wrapped in the conv template (which prepends a system
        message and the role prefix, and appends the assistant role tag).

        image_patch_block is one of:
          mm_use_im_start_end=True:  <im_start> <im_patch>*N <im_end>
          mm_use_im_start_end=False: <im_patch>*N

        For our merged VQA-RAD model: mm_use_im_start_end=True, N=256.
        """
        # Strip any pre-existing <image> token from the question; v1.0's
        # reference does the same so the placeholder isn't accidentally
        # left in after we replace it with patch tokens.
        qs = question.replace("<image>", "").strip()

        if self.mm_use_im_start_end:
            image_block = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len
                + DEFAULT_IM_END_TOKEN
            )
        else:
            image_block = DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len

        qs_with_image = qs + "\n" + image_block

        # Wrap in the conv template. .copy() is essential -- conv_templates
        # objects are shared singletons, and appending to them in-place
        # would leak state across samples.
        conv = self.conv_template.copy()
        conv.append_message(conv.roles[0], qs_with_image)

        # Append an empty assistant turn to signal "model, generate the
        # assistant's response now". This makes get_prompt() emit a
        # trailing "###Assistant:" that the model completes from.
        #
        # NOTE: v1.0's reference model_vqa_med.py does NOT do this -- a
        # confirmed bug in their published inference script that causes
        # every prediction to start with "Assistant: ". We intentionally
        # diverge for a cleaner baseline; see day-07 notes.
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        return prompt

    # -------------------------------------------------------------------------
    # Image loading + preprocessing
    # -------------------------------------------------------------------------

    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load + preprocess a single image into the model's expected tensor.

        Returns a (1, 3, H, W) tensor on CUDA in fp16.
        """
        image = Image.open(image_path).convert("RGB")
        # image_processor.preprocess(...) returns a dict; ["pixel_values"]
        # is a list (or batched tensor) of preprocessed images.
        # We take the single image and add a batch dim.
        image_tensor = self.image_processor.preprocess(
            image, return_tensors="pt"
        )["pixel_values"][0]
        return image_tensor.unsqueeze(0).half().cuda()

    # -------------------------------------------------------------------------
    # Per-sample generation
    # -------------------------------------------------------------------------

    def predict(self, sample) -> Prediction:
        """Generate an answer for one VQASample.

        Args:
            sample: a VQASample (must have .question, .image_path, .question_id)

        Returns:
            Prediction with .text being the model's decoded answer
            (truncated at the first '###' per v1.0's reference).
        """
        # --- Prompt + image tensor -----------------------------------------
        prompt = self._build_prompt(sample.question)
        images = self._load_image(sample.image_path)

        # --- Tokenize -------------------------------------------------------
        # We tokenize the full prompt as a single string; the model handles
        # the <im_patch> tokens internally by substituting CLIP embeddings
        # at those positions.
        inputs = self.tokenizer([prompt])
        input_ids = torch.as_tensor(inputs.input_ids).cuda()
        input_token_len = input_ids.shape[1]

        # --- Stopping criterion ---------------------------------------------
        # v1.0 stops on '###' (the conv template separator). The criterion
        # captures input_ids by reference; that's fine because input_ids is
        # not mutated during generation.
        keywords = ["###"]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        # --- Generate -------------------------------------------------------
        # do_sample=True, temperature=0.7, max_new_tokens=1024: byte-faithful
        # to model_vqa_med.py. Latency is measured wall-clock with CUDA
        # synchronisation (without sync, .time() returns before the GPU
        # actually finishes work, giving misleadingly fast numbers).
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria],
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # --- Sanity check: input portion of output should match input ------
        # v1.0's reference warns if any input token was modified, which
        # would indicate the model's image-token substitution went wrong.
        # We do the same.
        n_diff = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff > 0:
            # Don't crash -- the reference doesn't, and a single bad sample
            # shouldn't kill a 451-sample run. But surface it for review.
            print(
                f"[warn] {sample.question_id}: {n_diff} input tokens "
                f"differ from input portion of output_ids"
            )

        # --- Decode ---------------------------------------------------------
        # Slice off the input prompt; decode only the newly-generated tokens.
        # skip_special_tokens=True drops <im_patch>, <im_start>, <im_end>,
        # </s>, etc.
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        n_output_tokens = output_ids.shape[1] - input_token_len

        # --- Truncate at first conv.sep (i.e. '###') -----------------------
        # Even with KeywordsStoppingCriteria active, the stopping check
        # runs *after* the token is generated, so '###' may still be in
        # the decoded output. v1.0's reference trims it here.
        # If '###' isn't present (rare; model didn't emit it), append + find
        # to avoid a ValueError on .index().
        conv = self.conv_template.copy()
        try:
            index = outputs.index(conv.sep)
        except ValueError:
            outputs += conv.sep
            index = outputs.index(conv.sep)
        outputs = outputs[:index].strip()

        return Prediction(
            question_id=sample.question_id,
            text=outputs,
            prompt=sample.question,
            latency_ms=latency_ms,
            n_input_tokens=input_token_len,
            n_output_tokens=int(n_output_tokens),
        )

    # -------------------------------------------------------------------------
    # Batch entry point
    # -------------------------------------------------------------------------

    def run(
        self,
        samples: Sequence[Any],
        progress: bool = True,
        max_samples: Optional[int] = None,
    ) -> List[Prediction]:
        """Run inference over a list of samples.

        Args:
            samples: iterable of VQASample.
            progress: whether to print a tqdm-style progress bar.
            max_samples: if set, only run the first N. Useful for smoke
                         tests before launching the full eval.

        Returns:
            List of Prediction in the same order as samples.
        """
        items = list(samples)
        if max_samples is not None:
            items = items[:max_samples]

        if progress:
            try:
                from tqdm import tqdm
                items_iter = tqdm(items, desc="inference")
            except ImportError:
                items_iter = items
        else:
            items_iter = items

        predictions: List[Prediction] = []
        for sample in items_iter:
            pred = self.predict(sample)
            predictions.append(pred)

        return predictions