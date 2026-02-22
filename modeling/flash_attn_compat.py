import math
from typing import Optional

import torch

try:
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
except Exception:
    _flash_attn_varlen_func = None


def _expand_gqa_heads(tensor: torch.Tensor, target_heads: int) -> torch.Tensor:
    if tensor.size(0) == target_heads:
        return tensor
    if target_heads % tensor.size(0) != 0:
        raise ValueError(
            f"Incompatible head shapes for GQA: target={target_heads}, source={tensor.size(0)}"
        )
    repeat_factor = target_heads // tensor.size(0)
    return tensor.repeat_interleave(repeat_factor, dim=0)


def _fallback_flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    if cu_seqlens_q is None or cu_seqlens_k is None:
        raise ValueError("Fallback flash attention requires cu_seqlens_q and cu_seqlens_k.")

    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("Expected q, k, v to have shape [tokens, heads, dim].")

    if k.shape[2] != q.shape[2] or v.shape[2] != q.shape[2]:
        raise ValueError("Head dimension mismatch between q, k, v.")

    total_q, num_q_heads, head_dim = q.shape
    output = torch.empty_like(q)

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    q_boundaries = cu_seqlens_q.to(dtype=torch.int64, device="cpu")
    k_boundaries = cu_seqlens_k.to(dtype=torch.int64, device="cpu")

    if q_boundaries.numel() != k_boundaries.numel():
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same batch size.")

    num_samples = q_boundaries.numel() - 1
    for i in range(num_samples):
        q_start = q_boundaries[i].item()
        q_end = q_boundaries[i + 1].item()
        k_start = k_boundaries[i].item()
        k_end = k_boundaries[i + 1].item()

        if q_start == q_end:
            continue

        q_i = q[q_start:q_end].transpose(0, 1)
        k_i = k[k_start:k_end].transpose(0, 1)
        v_i = v[k_start:k_end].transpose(0, 1)

        if k_i.size(0) != num_q_heads:
            k_i = _expand_gqa_heads(k_i, num_q_heads)
            v_i = _expand_gqa_heads(v_i, num_q_heads)

        q_f = q_i.float()
        k_f = k_i.float()
        v_f = v_i.float()

        attn_scores = torch.matmul(q_f, k_f.transpose(-1, -2)) * softmax_scale

        if causal:
            q_len = q_i.shape[1]
            k_len = k_i.shape[1]
            # Align causal mask to the right for q_len != k_len, matching flash-attn behavior.
            offset = max(k_len - q_len, 0)
            q_pos = torch.arange(q_len, device=q.device).unsqueeze(-1) + offset
            k_pos = torch.arange(k_len, device=q.device).unsqueeze(0)
            causal_mask = k_pos <= q_pos
            attn_scores = attn_scores.masked_fill(
                ~causal_mask.unsqueeze(0),
                torch.finfo(attn_scores.dtype).min,
            )

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v_f).to(dtype=q.dtype)
        output[q_start:q_end] = attn_out.transpose(0, 1)

    if total_q != output.shape[0]:
        raise RuntimeError("Unexpected output shape in fallback flash attention.")
    return output


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor = None,
    cu_seqlens_k: torch.Tensor = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    **kwargs,
) -> torch.Tensor:
    if _flash_attn_varlen_func is not None and q.device.type == "cuda":
        return _flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
            softmax_scale=softmax_scale,
            **kwargs,
        )

    return _fallback_flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        causal=causal,
        softmax_scale=softmax_scale,
    )
