"""qwen_loader.py — Shared loader for Qwen audio model variants.

Centralises:
  - precision selection (4bit / 8bit / fp16 / bf16)
  - family detection (qwen2-audio  vs  qwen2.5-omni)
  - audio-tower attribute lookup (differs between families)
  - forward-pass-for-logits helper (Omni's `model.thinker(...)` vs Audio's `model(...)`)

Used by qwen_focus.py, qwenAudioInput.py, qwen_inspect.py so that all three
scripts can run any supported Qwen audio model at any supported precision
without code duplication.
"""

from __future__ import annotations

import torch
from transformers import AutoProcessor, BitsAndBytesConfig


# ---------------------------------------------------------------------------
# Family detection
# ---------------------------------------------------------------------------

def detect_family(model_name: str) -> str:
    """Return 'omni' for Qwen2.5-Omni* / Qwen3-Omni*, otherwise 'audio'
    (Qwen2-Audio* / Qwen-Audio*)."""
    n = model_name.lower()
    if "omni" in n:
        return "omni"
    return "audio"


def _detect_omni_subfamily(model_name: str) -> str:
    """Return 'qwen3' for Qwen3-Omni*, else 'qwen2_5' (Qwen2.5-Omni*)."""
    n = model_name.lower()
    if "qwen3" in n:
        return "qwen3"
    return "qwen2_5"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

VALID_PRECISIONS = ("4bit", "8bit", "fp16", "bf16")


def load_qwen(model_name: str, precision: str = "4bit"):
    """
    Load a Qwen audio model + processor at the requested precision.

    Returns
    -------
    model     : transformers PreTrainedModel (eval mode)
    processor : AutoProcessor
    target_sr : int (audio sample rate expected by the feature extractor)
    family    : "audio" | "omni"
    """
    if precision not in VALID_PRECISIONS:
        raise ValueError(
            f"precision must be one of {VALID_PRECISIONS}, got {precision!r}"
        )

    family = detect_family(model_name)

    quant_cfg = None
    dtype     = None
    if precision == "4bit":
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif precision == "8bit":
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "fp16":
        dtype = torch.float16
    elif precision == "bf16":
        dtype = torch.bfloat16

    print(
        f"Loading {model_name}  (precision={precision}, family={family}) …",
        flush=True,
    )

    if family == "omni":
        sub = _detect_omni_subfamily(model_name)
        if sub == "qwen3":
            # Qwen3-Omni MoE (Qwen3-Omni-30B-A3B-*). Requires a recent
            # transformers (>= 4.57 / main) that ships Qwen3OmniMoe classes.
            try:
                from transformers import (
                    Qwen3OmniMoeForConditionalGeneration as ModelCls,
                )
            except ImportError as e:
                raise ImportError(
                    "Qwen3-Omni requires transformers with Qwen3OmniMoe "
                    "support (>= 4.57 or install from main). "
                    "Upgrade with: pip install -U "
                    "'transformers>=4.57' accelerate"
                ) from e
        else:
            # Qwen2.5-Omni (available in transformers >= 4.50)
            try:
                from transformers import (
                    Qwen2_5OmniForConditionalGeneration as ModelCls,
                )
            except ImportError as e:
                raise ImportError(
                    "Qwen2.5-Omni requires transformers>=4.50. "
                    "Upgrade with: pip install -U transformers accelerate"
                ) from e
    else:
        from transformers import Qwen2AudioForConditionalGeneration as ModelCls

    kwargs = {"device_map": {"": 0}}
    if quant_cfg is not None:
        kwargs["quantization_config"] = quant_cfg
    if dtype is not None:
        kwargs["torch_dtype"] = dtype

    processor = AutoProcessor.from_pretrained(model_name)
    model     = ModelCls.from_pretrained(model_name, **kwargs)
    model.eval()

    # For Omni: drop the speech-generation half (talker + token2wav).
    # We only need text/audio understanding; the TTS half costs ~6GB VRAM
    # and is invoked by default in .generate(), which causes OOM on a 24GB GPU.
    if family == "omni":
        if hasattr(model, "disable_talker"):
            try:
                model.disable_talker()
            except Exception as e:
                print(f"  (disable_talker() failed: {e})")
        # Manual cleanup in case disable_talker did not exist or did not free memory
        for attr in ("talker", "token2wav"):
            if hasattr(model, attr):
                try:
                    setattr(model, attr, None)
                except Exception:
                    pass
        torch.cuda.empty_cache()

    # Most processors expose feature_extractor.sampling_rate; fall back to 16k.
    target_sr = 16000
    fe = getattr(processor, "feature_extractor", None)
    if fe is not None and hasattr(fe, "sampling_rate"):
        target_sr = fe.sampling_rate

    print("Model ready.\n", flush=True)
    return model, processor, target_sr, family


# ---------------------------------------------------------------------------
# Family-aware accessors (used by qwen_inspect.py)
# ---------------------------------------------------------------------------

def get_audio_tower(model, family: str):
    """Return the audio encoder submodule for this model family."""
    if family == "omni":
        # Qwen2.5-Omni has a thinker/talker split; the audio encoder lives in thinker.
        if hasattr(model, "thinker") and hasattr(model.thinker, "audio_tower"):
            return model.thinker.audio_tower
        if hasattr(model, "audio_tower"):
            return model.audio_tower
        raise AttributeError(
            "Could not locate audio_tower on Qwen2.5-Omni model "
            "(expected model.thinker.audio_tower or model.audio_tower)."
        )
    # Qwen2-Audio
    return model.audio_tower


def forward_for_logits(model, family: str, inputs):
    """
    Run a forward pass and return text-vocab logits of shape (B, T, V).

    For Qwen2-Audio:  call model(**inputs) and use .logits
    For Qwen2.5-Omni: call model.thinker(**inputs) and use .logits
                      (talker generates speech, which we do not want here)
    """
    # Cast input_features to the audio tower's parameter dtype if needed,
    # so the first Conv layer (which may be bf16/fp16) accepts it.
    if "input_features" in inputs and inputs["input_features"] is not None:
        try:
            audio_tower = get_audio_tower(model, family)
            tower_dtype = next(audio_tower.parameters()).dtype
            if inputs["input_features"].dtype != tower_dtype:
                inputs = dict(inputs)
                inputs["input_features"] = inputs["input_features"].to(tower_dtype)
        except (AttributeError, StopIteration):
            pass
    if family == "omni" and hasattr(model, "thinker"):
        out = model.thinker(**inputs)
    else:
        out = model(**inputs)
    if hasattr(out, "logits"):
        return out.logits
    # Some forward returns may name it differently
    for attr in ("thinker_logits", "text_logits"):
        if hasattr(out, attr):
            return getattr(out, attr)
    raise AttributeError(
        "Forward output has no .logits / .thinker_logits attribute."
    )


def encode_audio(model, family: str, inputs, output_hidden_states: bool = False):
    """
    Run the audio encoder for either Qwen2-Audio or Qwen2.5-Omni and return
    the raw transformer output (with `.last_hidden_state` and optionally
    `.hidden_states`).

    Handles the differing forward signatures:
      Qwen2-Audio:   audio_tower(input_features, feature_attention_mask=...)
      Qwen2.5-Omni:  audio_tower(input_features, feature_lens=..., ...)
                     where feature_lens = feature_attention_mask.sum(-1)
    """
    import torch as _torch
    audio_tower = get_audio_tower(model, family)
    input_features = inputs["input_features"].to(model.device)
    # Cast to the audio tower's parameter dtype so the first Conv accepts it.
    # (Qwen3-Omni's conv2d1 is strict; Qwen2-Audio / Qwen2.5-Omni tolerate fp32
    # input even when loaded in fp16/bf16, but this cast is safe for all of them.)
    try:
        tower_dtype = next(audio_tower.parameters()).dtype
        if input_features.dtype != tower_dtype:
            input_features = input_features.to(tower_dtype)
    except StopIteration:
        pass
    mask = inputs.get("feature_attention_mask")

    kwargs = {"output_hidden_states": output_hidden_states}

    if family == "omni":
        # Omni's audio_tower expects:
        #   input_features:  (n_mels, sum(feature_lens))   <- flat, NO padding
        #   feature_lens:    1-D long tensor of valid frame counts
        # The padded (B, n_mels, T_max) tensor from the processor must be
        # trimmed using feature_attention_mask and concatenated along T.
        if mask is not None:
            mask_bool = mask.to(model.device).bool()
            feature_lens = mask_bool.sum(dim=-1).long()
            # (B, n_mels, T) -> (B, T, n_mels) -> select valid frames
            # -> (total_valid, n_mels) -> (n_mels, total_valid)
            input_features = (
                input_features.permute(0, 2, 1)[mask_bool].permute(1, 0)
            )
        else:
            # No mask -> assume full length; flatten batch into time dim
            B, M, T = input_features.shape
            feature_lens = _torch.tensor(
                [T] * B, device=model.device, dtype=_torch.long,
            )
            input_features = input_features.permute(1, 0, 2).reshape(M, B * T)
        kwargs["feature_lens"] = feature_lens
    else:
        if mask is not None:
            kwargs["feature_attention_mask"] = mask

    with _torch.no_grad():
        return audio_tower(input_features, **kwargs)


def generate_text_only(model, family: str, **gen_kwargs):
    """
    Wrapper around model.generate() that disables speech generation for Omni.
    For Qwen2-Audio it is a straight passthrough.
    """
    if family == "omni":
        # Qwen2.5-Omni generate() accepts return_audio=False to skip TTS
        gen_kwargs.setdefault("return_audio", False)
    # Cast input_features to the audio tower's parameter dtype if needed.
    if "input_features" in gen_kwargs and gen_kwargs["input_features"] is not None:
        try:
            audio_tower = get_audio_tower(model, family)
            tower_dtype = next(audio_tower.parameters()).dtype
            if gen_kwargs["input_features"].dtype != tower_dtype:
                gen_kwargs["input_features"] = gen_kwargs["input_features"].to(tower_dtype)
        except (AttributeError, StopIteration):
            pass
    return model.generate(**gen_kwargs)
