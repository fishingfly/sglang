import torch

def musa_batched_rotary_embedding_contiguous(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    rot_dim: int,
    cos_sin_cache_offsets: torch.Tensor,
)-> None:
    return torch.ops.sgl_kernel.musa_batched_rotary_embedding_contiguous(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox,
        rot_dim,
        cos_sin_cache_offsets,
    )

def musa_rotary_embedding_contiguous(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
)-> None:
    return torch.ops.sgl_kernel.musa_rotary_embedding_contiguous(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox,
    )