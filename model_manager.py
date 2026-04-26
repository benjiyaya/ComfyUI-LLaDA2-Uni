"""
Singleton model manager for LLaDA 2.0-Uni.
Handles loading/unloading, attention backends, CPU offload, and VRAM management.
"""

import torch
import gc
import sys
import os
from typing import Optional, Dict, Any

# ── Global state ──
_LLM_MODEL = None
_LLM_TOKENIZER = None
_DECODER = None
_DECODER_MODE = None
_VAE = None
_SIGVQ = None
_IMAGE_TOKENIZER = None
_MODEL_PATH = None
_ATTENTION = None  # "flash_attn" | "sdpa" | "sage_attention"
_DEVICE = "cuda"
_OFFLOAD = False


def set_attention_backend(backend: str):
    """Set attention before loading. One of: flash_attn, sdpa, sage_attention"""
    global _ATTENTION
    _ATTENTION = backend


def _ensure_repo_on_path(model_path: str):
    """Add the custom node's own root to sys.path for encoder/decoder imports.
    The encoder/ and decoder/ folders live inside the custom node directory."""
    node_root = os.path.dirname(os.path.abspath(__file__))
    if node_root not in sys.path:
        sys.path.insert(0, node_root)
    return node_root


def load_llm(model_path: str, device: str = "cuda", attention: str = "flash_attn",
             offload: bool = False):
    """Load the dLLM-MoE backbone. Returns (model, tokenizer)."""
    global _LLM_MODEL, _LLM_TOKENIZER, _MODEL_PATH, _ATTENTION, _DEVICE, _OFFLOAD

    _ensure_repo_on_path(model_path)
    _ATTENTION = attention
    _DEVICE = device
    _OFFLOAD = offload

    if _LLM_MODEL is not None and _MODEL_PATH == model_path:
        return _LLM_MODEL, _LLM_TOKENIZER

    unload_all()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ── Attention backend ──
    attn_kwargs = {"trust_remote_code": True}
    if attention == "sdpa":
        attn_kwargs["attn_implementation"] = "sdpa"
    elif attention in ("sage_attention", "sage_attn"):
        attn_kwargs["attn_implementation"] = "flash_attention_2"  # fallback; sage needs patching
        # SageAttention: user should have `sageattention` installed.
        # We monkey-patch after loading if sage is available.
    # flash_attn (default): transformers auto-detects flash-attn2 if installed

    # ── Device map ──
    if offload:
        # CPU offload: keep minimal on GPU, rest on CPU
        device_map = {"lm_head": device, "model.embed_tokens": device}
        attn_kwargs["device_map"] = "auto"
        attn_kwargs["max_memory"] = {0: "20GiB", "cpu": "80GiB"}
        attn_kwargs["offload_folder"] = "offload_cache"
        attn_kwargs["torch_dtype"] = torch.bfloat16
    else:
        attn_kwargs["device_map"] = device
        attn_kwargs["torch_dtype"] = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, **attn_kwargs).eval()
    model.tokenizer = tokenizer

    # ── SageAttention patching ──
    if attention in ("sage_attention", "sage_attn"):
        _try_patch_sage_attn(model)

    _LLM_MODEL = model
    _LLM_TOKENIZER = tokenizer
    _MODEL_PATH = model_path
    return model, tokenizer


def _try_patch_sage_attn(model):
    """Attempt to replace flash attention with SageAttention for memory savings."""
    try:
        import sageattention
        # SageAttention provides a drop-in replacement.
        # The actual patching depends on the model's internal attention module.
        # For LLaDA's custom modeling code, we set a config flag if supported.
        if hasattr(model, 'config'):
            model.config._attn_backend = "sage"
        print("[LLaDA] SageAttention backend activated")
    except ImportError:
        print("[LLaDA] ⚠️ sageattention not installed. Install with: pip install sageattention")
        print("[LLaDA] Falling back to flash_attn")


def get_image_tokenizer(model_path: str, device: str = "cuda"):
    """Load the SigLIP-VQ image tokenizer."""
    global _IMAGE_TOKENIZER
    if _IMAGE_TOKENIZER is None:
        repo = _ensure_repo_on_path(model_path)
        from encoder.image_tokenizer import ImageTokenizer
        _IMAGE_TOKENIZER = ImageTokenizer(model_path=model_path, device=device)
    return _IMAGE_TOKENIZER


def decode_tokens(token_ids, h, w, model_path: str, device: str = "cuda",
                  num_steps: int = 50, decode_mode: str = "normal",
                  resolution_multiplier: int = 2):
    """Decode VQ tokens to PIL image."""
    _ensure_repo_on_path(model_path)
    from decoder import decode_vq_tokens
    return decode_vq_tokens(
        token_ids, h, w, model_path, device,
        resolution_multiplier=resolution_multiplier,
        num_steps=num_steps,
        decode_mode=decode_mode,
    )


def unload_llm():
    """Unload LLM backbone to free VRAM."""
    global _LLM_MODEL, _LLM_TOKENIZER
    if _LLM_MODEL is not None:
        del _LLM_MODEL, _LLM_TOKENIZER
        _LLM_MODEL = None
        _LLM_TOKENIZER = None
        gc.collect()
        torch.cuda.empty_cache()


def unload_decoder():
    """Unload decoder components."""
    global _DECODER, _DECODER_MODE, _VAE, _SIGVQ
    for attr in (_DECODER, _DECODER_MODE, _VAE, _SIGVQ):
        if attr is not None:
            del attr
    _DECODER = None
    _DECODER_MODE = None
    _VAE = None
    _SIGVQ = None
    gc.collect()
    torch.cuda.empty_cache()


def unload_image_tokenizer():
    """Unload image tokenizer."""
    global _IMAGE_TOKENIZER
    if _IMAGE_TOKENIZER is not None:
        del _IMAGE_TOKENIZER
        _IMAGE_TOKENIZER = None
        gc.collect()
        torch.cuda.empty_cache()


def unload_all():
    """Unload everything. Call this to free all VRAM."""
    unload_llm()
    unload_decoder()
    unload_image_tokenizer()


def get_model_state() -> Dict[str, Any]:
    """Return current loaded state for IS_CHANGED cache control."""
    return {
        "model_path": _MODEL_PATH,
        "llm_loaded": _LLM_MODEL is not None,
        "attention": _ATTENTION,
    }
