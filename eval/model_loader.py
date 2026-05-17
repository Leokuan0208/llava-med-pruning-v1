"""LLaVA-Med v1.0 model loader.

The v1.0 codebase does NOT expose a clean `load_pretrained_model`
helper the way v1.5 does. Loading is hand-rolled in model_vqa_med.py
as a ~50-line init sequence; this file mirrors that sequence faithfully
so the runner can stay agnostic to model-version internals.

Key differences vs. the v1.5 loader:

  - Direct construction: AutoTokenizer + LlavaLlamaForCausalLM, no
    builder helper.
  - Special tokens added at load time: <im_patch>, <im_start>, <im_end>.
    The tokenizer files on disk include them in added_tokens.json, but
    the v1.0 inference script ADDS them again explicitly anyway -- we
    mirror that to stay byte-faithful.
  - Image processor built separately from CLIPImageProcessor against
    model.config.mm_vision_tower (which the merged model has set to
    "openai/clip-vit-large-patch14").
  - image_token_len computed from vision_config:
      (image_size // patch_size)^2  =  (224/14)^2  =  256.
    This is HALF of v1.5's 576 -- meaningful for downstream pruning
    work, but for now it just means the prompt has 256 patch tokens.
  - conv_mode is "simple" (text-only chat template; the image is
    encoded via patch tokens in the prompt rather than via a system
    prompt).
"""

import os
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel

from llava import LlavaLlamaForCausalLM
from llava.utils import disable_torch_init


# Token strings used by LLaVA-Med v1.0. Reproduced verbatim from
# model_vqa_med.py so the constants don't drift if we ever pin a
# different LLaVA-Med revision.
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


@dataclass
class LoadedModel:
    """Everything the runner needs to generate from a v1.0 VLM.

    Bundling these together means the runner has one object to pass
    around instead of six.
    """
    model: Any                # LlavaLlamaForCausalLM on cuda, float16
    tokenizer: Any            # AutoTokenizer with image patch tokens added
    image_processor: Any      # CLIPImageProcessor for the vision tower
    context_len: int          # Max sequence length the model supports
    conv_mode: str            # Conversation template name ("simple")
    image_token_len: int      # 256 for CLIP-ViT-L/14 at 224x224
    mm_use_im_start_end: bool # Whether to wrap patch tokens in <im_start>/<im_end>


def load_llava_med_v1(
    model_path: str,
    conv_mode: str = "simple",
    device: str = "cuda",
) -> LoadedModel:
    """Load a LLaVA-Med v1.0 model (or one of its dataset-finetuned variants).

    Args:
        model_path: Filesystem path to a HuggingFace-format model dir
            produced by `python -m llava.model.apply_delta`. For example,
            /data/dan/weights/llava-med-7b-vqarad-merged.
        conv_mode: Conversation template name. Defaults to "simple", which
            matches the default in v1.0's model_vqa_med.py and is the
            template that produced the paper numbers.
        device: Where to place the model. Almost always "cuda".

    Returns:
        LoadedModel bundle.

    Raises:
        FileNotFoundError: if model_path does not exist on disk.
    """
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Model path does not exist or is not a directory: {model_path}"
        )

    # disable_torch_init() skips some default torch initialization that
    # would be overwritten by from_pretrained anyway. It's a small startup
    # speedup borrowed verbatim from model_vqa_med.py.
    disable_torch_init()

    # --- Tokenizer ----------------------------------------------------------
    # AutoTokenizer rather than LlamaTokenizer specifically because the
    # tokenizer files on disk may or may not be the slow LlamaTokenizer
    # depending on which transformers version produced them; AutoTokenizer
    # picks the right class automatically.
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # --- Model --------------------------------------------------------------
    # float16 inference; .cuda() moves to GPU. use_cache=True enables KV
    # cache for generation (we generate auto-regressively, so cache helps).
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        use_cache=True,
    ).to(device)

    # --- Image processor ---------------------------------------------------
    # Built from the vision tower path stored in the model config. For our
    # merged model this is "openai/clip-vit-large-patch14" -- HuggingFace
    # will download the processor config the first time (cached afterward).
    image_processor = CLIPImageProcessor.from_pretrained(
        model.config.mm_vision_tower,
        torch_dtype=torch.float16,
    )

    # Move the model's internal vision tower to the right device + dtype.
    # The vision tower is stored as a single-element list inside model.model
    # (LLaVA's pattern); we reach in and set it explicitly.
    vision_tower = model.model.vision_tower[0]
    vision_tower.to(device=device, dtype=torch.float16)

    # --- Register the image patch tokens -----------------------------------
    # The tokenizer files on disk have these in added_tokens.json, so this
    # call is mostly a no-op. We do it anyway because model_vqa_med.py does,
    # and skipping it might silently leave the special_tokens attribute
    # incomplete in subtle ways.
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN],
            special_tokens=True,
        )

    # --- Wire the vision config -------------------------------------------
    # The model needs to know which token ID corresponds to <im_patch>
    # so it can find the patch positions in the input and inject the
    # CLIP embeddings there. This is a v1.0-specific wiring step that
    # model_vqa_med.py does inline; we mirror it.
    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_IMAGE_PATCH_TOKEN]
    )[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        (
            vision_config.im_start_token,
            vision_config.im_end_token,
        ) = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
        )

    # --- Compute image_token_len ------------------------------------------
    # (image_size // patch_size)^2 visual tokens per image.
    # For CLIP-ViT-L/14 at 224x224: (224/14)^2 = 16^2 = 256.
    # This is what the prompt will use to repeat <im_patch>.
    image_token_len = (
        vision_config.image_size // vision_config.patch_size
    ) ** 2

    # --- Context length ---------------------------------------------------
    # v1.0 doesn't expose a clean context_len; we read max_sequence_length
    # from config (set to 2048 in our merged model).
    context_len = getattr(model.config, "max_sequence_length", 2048)

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        context_len=context_len,
        conv_mode=conv_mode,
        image_token_len=image_token_len,
        mm_use_im_start_end=mm_use_im_start_end,
    )