"""Equivalent grouped-query attention with explicit KV repetition for older GPUs.

Uses Transformers' public attention/mask registries. No CUDA kernels or model weights
are modified; PyTorch still computes full attention, with the original causal/padding mask.
"""


def repeated_kv_sdpa(module, query, key, value, attention_mask, dropout=0.0,
                     scaling=None, is_causal=None, position_bias=None, **kwargs):
    import torch.nn.functional as F
    if position_bias is not None or kwargs.get("output_attentions", False):
        raise ValueError("This embedding-only adapter does not implement position bias or attention outputs")
    if query.shape[1] % key.shape[1] or key.shape[1] != value.shape[1]:
        raise ValueError("Incompatible attention head counts")
    repeats = query.shape[1] // key.shape[1]
    if repeats > 1:
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
    causal = getattr(module, "is_causal", True) if is_causal is None else is_causal
    causal = bool(causal and attention_mask is None and query.shape[2] > 1)
    result = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=dropout,
        is_causal=causal, scale=scaling,
    )
    return result.transpose(1, 2).contiguous(), None


def register_repeated_kv_attention():
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import sdpa_mask
    AttentionInterface.register("batill_sdpa_repeat_kv", repeated_kv_sdpa)
    AttentionMaskInterface.register("batill_sdpa_repeat_kv", sdpa_mask)
