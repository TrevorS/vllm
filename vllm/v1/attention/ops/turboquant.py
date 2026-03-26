# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant KV cache compression.

Implements the TurboQuant algorithm (Google, ICLR 2026) for compressing
key vectors in the KV cache via random orthogonal rotation + Lloyd-Max
scalar quantization.

Reference: https://arxiv.org/abs/2504.19874
"""

import math
from functools import lru_cache

import torch
from scipy import integrate as sp_integrate
from scipy import special as sp_special

# QJL correction constant: sqrt(pi/2)
QJL_SCALE = math.sqrt(math.pi / 2.0)

# Precomputed bit-packing weights (graph-safe: no tensor creation during forward)
_BIT_PACK_WEIGHTS: dict[torch.device, torch.Tensor] = {}


def _pack_sign_bits(bits: torch.Tensor) -> torch.Tensor:
    """Pack [..., 8] uint8 bit tensor into [...] uint8 bytes. Graph-safe."""
    device = bits.device
    if device not in _BIT_PACK_WEIGHTS:
        _BIT_PACK_WEIGHTS[device] = torch.tensor(
            [128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=device
        )
    weights = _BIT_PACK_WEIGHTS[device]
    return (bits * weights).sum(dim=-1).to(torch.uint8)


# ---------------------------------------------------------------------------
# Precomputation: rotation matrices and Lloyd-Max centroids
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _build_rotation_matrix(
    head_dim: int,
    device: torch.device,
    seed: int = 42,
) -> torch.Tensor:
    """Build a deterministic random orthogonal matrix via QR decomposition.

    Returns:
        R: [head_dim, head_dim] orthogonal matrix, float32
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    random_matrix = torch.randn(head_dim, head_dim, generator=gen)
    q, r = torch.linalg.qr(random_matrix)
    # Ensure deterministic sign (Haar measure correction)
    diag_sign = torch.sign(torch.diag(r))
    q = q * diag_sign.unsqueeze(0)
    return q.to(device=device, dtype=torch.float32)


def _beta_pdf(x: float, d: int) -> float:
    """PDF of the marginal distribution of a coordinate on the unit sphere
    in R^d after random rotation.

    This is Beta((d-1)/2, (d-1)/2) scaled to [-1, 1].
    """
    if abs(x) >= 1.0:
        return 0.0
    a = (d - 1) / 2.0
    # Normalization: Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2))
    norm = math.exp(
        sp_special.gammaln(d / 2.0)
        - 0.5 * math.log(math.pi)
        - sp_special.gammaln((d - 1) / 2.0)
    )
    return norm * (1.0 - x * x) ** (a - 1)


@lru_cache(maxsize=8)
def _compute_lloyd_max_centroids(
    head_dim: int,
    num_bits: int,
    num_iterations: int = 300,
) -> tuple[list[float], list[float]]:
    """Compute Lloyd-Max optimal centroids for the Beta distribution
    arising from random rotation of unit sphere vectors in R^head_dim.

    Returns:
        centroids: sorted list of 2^num_bits centroid values
        boundaries: sorted list of 2^num_bits - 1 boundary values
    """
    num_centroids = 1 << num_bits
    d = head_dim

    # Initialize centroids uniformly in the support.
    # The effective support of the Beta distribution on [-1, 1] narrows
    # as d increases. Use +/-3/sqrt(d) as practical bounds.
    support = 3.0 / math.sqrt(d) if d > 10 else 0.99
    centroids = [
        -support + (2 * support) * (i + 0.5) / num_centroids
        for i in range(num_centroids)
    ]

    for _ in range(num_iterations):
        # Update boundaries: midpoints between consecutive centroids
        boundaries = [
            (centroids[i] + centroids[i + 1]) / 2.0 for i in range(num_centroids - 1)
        ]

        # Update centroids: conditional expectation within each partition
        edges = [-1.0] + boundaries + [1.0]
        new_centroids = []
        for i in range(num_centroids):
            lo, hi = edges[i], edges[i + 1]
            # E[X | lo <= X <= hi] = integral(x * pdf(x)) / integral(pdf(x))
            num, _ = sp_integrate.quad(lambda x: x * _beta_pdf(x, d), lo, hi)
            den, _ = sp_integrate.quad(lambda x: _beta_pdf(x, d), lo, hi)
            if den > 1e-15:
                new_centroids.append(num / den)
            else:
                new_centroids.append((lo + hi) / 2.0)
        centroids = new_centroids

    boundaries = [
        (centroids[i] + centroids[i + 1]) / 2.0 for i in range(num_centroids - 1)
    ]
    return centroids, boundaries


@lru_cache(maxsize=8)
def get_turboquant_params(
    head_dim: int,
    num_bits: int,
    device: torch.device,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get precomputed TurboQuant parameters.

    Returns:
        rotation_matrix: [head_dim, head_dim] float32
        centroids: [2^num_bits] float32
        boundaries: [2^num_bits - 1] float32
    """
    rotation_matrix = _build_rotation_matrix(head_dim, device, seed)
    centroid_vals, boundary_vals = _compute_lloyd_max_centroids(head_dim, num_bits)
    centroids = torch.tensor(centroid_vals, dtype=torch.float32, device=device)
    boundaries = torch.tensor(boundary_vals, dtype=torch.float32, device=device)
    return rotation_matrix, centroids, boundaries


def num_bits_from_dtype_str(kv_cache_dtype: str) -> int:
    """Extract bit width from cache dtype string."""
    if kv_cache_dtype == "tq3":
        return 3
    elif kv_cache_dtype in ("tq4", "tq4o"):
        return 4
    else:
        raise ValueError(f"Unknown TurboQuant dtype: {kv_cache_dtype}")


# ---------------------------------------------------------------------------
# Quantize / Dequantize
# ---------------------------------------------------------------------------


def quantize_keys(
    keys: torch.Tensor,
    rotation_matrix: torch.Tensor,
    boundaries: torch.Tensor,
    centroids: torch.Tensor | None = None,
    compute_qjl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Quantize key vectors using TurboQuant.

    Args:
        keys: [num_tokens, num_kv_heads, head_dim] in any float dtype
        rotation_matrix: [head_dim, head_dim] float32 orthogonal matrix
        boundaries: [2^b - 1] float32 sorted boundary values
        centroids: [2^b] float32 -- needed if compute_qjl=True
        compute_qjl: if True, also compute QJL sign bits + residual norms

    Returns:
        indices: [num_tokens, num_kv_heads, head_dim] uint8
        norms: [num_tokens, num_kv_heads] float16
        qjl_sign_bits: [num_tokens, num_kv_heads, head_dim // 8] uint8
            (or None)
        qjl_residual_norms: [num_tokens, num_kv_heads] float16 (or None)
    """
    # Compute norms and normalize
    norms = keys.float().norm(dim=-1)  # [N, H]
    safe_norms = norms.clamp(min=1e-8)
    keys_unit = keys.float() / safe_norms.unsqueeze(-1)  # [N, H, D]

    # Rotate: keys_unit @ R^T
    rotated = torch.matmul(keys_unit, rotation_matrix.t())  # [N, H, D]

    # Quantize via searchsorted
    indices = torch.searchsorted(boundaries, rotated.reshape(-1)).reshape(rotated.shape)
    indices = indices.to(torch.uint8)

    qjl_sign_bits = None
    qjl_residual_norms = None

    if compute_qjl and centroids is not None:
        # QJL Stage 2: compute residual in rotated space, store sign + norm
        rotated_approx = centroids[indices.long()]  # [N, H, D]
        residual = rotated - rotated_approx  # [N, H, D]

        # Sign bits: pack (residual >= 0) into bytes
        sign_positive = (residual >= 0).to(torch.uint8)  # [N, H, D]
        D = sign_positive.shape[-1]
        bits = sign_positive.reshape(*sign_positive.shape[:-1], D // 8, 8)
        qjl_sign_bits = _pack_sign_bits(bits)

        # Residual norms (in rotated space, on unit-normalized key)
        qjl_residual_norms = residual.norm(dim=-1).to(torch.float16)

    return indices, norms.to(torch.float16), qjl_sign_bits, qjl_residual_norms


def dequantize_keys(
    indices: torch.Tensor,
    norms: torch.Tensor,
    rotation_matrix: torch.Tensor,
    centroids: torch.Tensor,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize key vectors from TurboQuant indices.

    Args:
        indices: [num_tokens, num_kv_heads, head_dim] uint8
        norms: [num_tokens, num_kv_heads] float16
        rotation_matrix: [head_dim, head_dim] float32 orthogonal matrix
        centroids: [2^b] float32 centroid values
        target_dtype: output dtype (default bfloat16)

    Returns:
        keys_approx: [num_tokens, num_kv_heads, head_dim] in target_dtype
    """
    # Lookup centroids
    rotated_approx = centroids[indices.long()]  # [N, H, D] float32

    # Unrotate: rotated_approx @ R  (inverse rotation = transpose)
    keys_approx = torch.matmul(rotated_approx, rotation_matrix)  # [N, H, D]

    # Scale by norms
    keys_approx = keys_approx * norms.float().unsqueeze(-1)

    return keys_approx.to(target_dtype)


# ---------------------------------------------------------------------------
# TurboQuantCache -- split K/V/norm container
# ---------------------------------------------------------------------------


class TurboQuantCache:
    """Container for split TQ key/value/norm cache tensors.

    Replaces the single kv_cache tensor for TurboQuant layers.
    Only triton_attn backend knows how to unwrap this -- all other
    backends never see it (TQ is only in triton_attn's supported list).

    """

    __slots__ = (
        "key_indices",
        "norms",
        "value_cache",
        "num_bits",
        "qjl_signs",
        "qjl_residual_norms",
        "outlier_cache",
        "normal_norms",
        "use_outliers",
    )

    def __init__(
        self,
        key_indices: torch.Tensor,
        norms: torch.Tensor,
        value_cache: torch.Tensor,
        num_bits: int = 4,
        qjl_signs: torch.Tensor | None = None,
        qjl_residual_norms: torch.Tensor | None = None,
        outlier_cache: torch.Tensor | None = None,
        normal_norms: torch.Tensor | None = None,
        use_outliers: bool = False,
    ):
        self.key_indices = key_indices
        self.norms = norms
        self.value_cache = value_cache
        self.num_bits = num_bits
        self.qjl_signs = qjl_signs
        self.qjl_residual_norms = qjl_residual_norms
        self.outlier_cache = outlier_cache
        self.normal_norms = normal_norms
        self.use_outliers = use_outliers

    @property
    def device(self) -> torch.device:
        return self.key_indices.device

    @property
    def dtype(self) -> torch.dtype:
        return self.value_cache.dtype


def prerotate_queries(
    queries: torch.Tensor,
    rotation_matrix: torch.Tensor,
) -> torch.Tensor:
    """Pre-rotate query vectors for fused TQ attention.

    Instead of rotating every cached key back (O(seq_len * D^2)),
    rotate the query once (O(D^2)) and dot with centroids directly.

    Identity: <q, R^T * c[idx]> = <R*q, c[idx]>

    Args:
        queries: [num_tokens, num_heads, head_dim]
        rotation_matrix: [head_dim, head_dim] float32 orthogonal

    Returns:
        queries_rotated: [num_tokens, num_heads, head_dim] same dtype
    """
    dtype = queries.dtype
    # R*q = queries @ R^T
    q_rot = torch.matmul(queries.float(), rotation_matrix.t())
    return q_rot.to(dtype)


# ---------------------------------------------------------------------------
# Paged cache helpers
# ---------------------------------------------------------------------------


def _pack_3bit_group(indices: torch.Tensor) -> torch.Tensor:
    """Pack 3-bit indices into bytes (8 indices -> 3 bytes).

    Args:
        indices: [..., D] uint8 where D is divisible by 8, values 0-7

    Returns:
        packed: [..., D * 3 // 8] uint8
    """
    shape = indices.shape
    D = shape[-1]
    assert D % 8 == 0, f"Dimension {D} must be divisible by 8 for 3-bit packing"

    # Reshape to [..., D//8, 8] for group processing
    idx = indices.reshape(*shape[:-1], D // 8, 8).to(torch.int32)
    i0, i1, i2, i3, i4, i5, i6, i7 = idx.unbind(-1)

    # Pack 8 x 3-bit values into 3 bytes (24 bits)
    byte0 = ((i0 << 5) | (i1 << 2) | (i2 >> 1)).to(torch.uint8)
    byte1 = (((i2 & 1) << 7) | (i3 << 4) | (i4 << 1) | (i5 >> 2)).to(torch.uint8)
    byte2 = (((i5 & 3) << 6) | (i6 << 3) | i7).to(torch.uint8)

    # Stack and flatten: [..., D//8, 3] -> [..., D*3//8]
    packed = torch.stack([byte0, byte1, byte2], dim=-1)
    return packed.reshape(*shape[:-1], D * 3 // 8)


def _unpack_3bit_group(packed: torch.Tensor, num_indices: int) -> torch.Tensor:
    """Unpack 3-bit packed bytes back to indices (3 bytes -> 8 indices).

    Args:
        packed: [..., num_indices * 3 // 8] uint8
        num_indices: number of indices to unpack

    Returns:
        indices: [..., num_indices] uint8, values 0-7
    """
    shape = packed.shape
    num_groups = num_indices // 8
    p = packed.reshape(*shape[:-1], num_groups, 3).to(torch.int32)
    b0, b1, b2 = p.unbind(-1)

    i0 = (b0 >> 5) & 7
    i1 = (b0 >> 2) & 7
    i2 = ((b0 & 3) << 1) | ((b1 >> 7) & 1)
    i3 = (b1 >> 4) & 7
    i4 = (b1 >> 1) & 7
    i5 = ((b1 & 1) << 2) | ((b2 >> 6) & 3)
    i6 = (b2 >> 3) & 7
    i7 = b2 & 7

    indices = torch.stack([i0, i1, i2, i3, i4, i5, i6, i7], dim=-1)
    return indices.reshape(*shape[:-1], num_indices).to(torch.uint8)


def quantize_and_store(
    keys: torch.Tensor,
    key_cache: torch.Tensor,
    norm_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rotation_matrix: torch.Tensor,
    boundaries: torch.Tensor,
    block_size: int,
    num_bits: int = 4,
    centroids: torch.Tensor | None = None,
    qjl_sign_cache: torch.Tensor | None = None,
    qjl_rnorm_cache: torch.Tensor | None = None,
) -> None:
    """Quantize keys and store into paged cache buffers.

    CUDA-graph-safe: uses clamp + scatter instead of boolean indexing.

    Args:
        keys: [num_tokens, num_kv_heads, head_dim]
        key_cache: [num_blocks, block_size, num_kv_heads, packed_dim] uint8
        norm_cache: [num_blocks, block_size, num_kv_heads] float16
        slot_mapping: [num_tokens] int64
        rotation_matrix: [head_dim, head_dim] float32
        boundaries: [2^b - 1] float32
        block_size: page block size
        num_bits: 3 or 4 -- determines packing scheme
        centroids: needed for QJL residual computation
        qjl_sign_cache: [num_blocks, block_size, num_kv_heads, D//8] uint8
        qjl_rnorm_cache: [num_blocks, block_size, num_kv_heads] float16
    """
    compute_qjl = qjl_sign_cache is not None
    indices, norms, qjl_signs, qjl_rnorms = quantize_keys(
        keys,
        rotation_matrix,
        boundaries,
        centroids=centroids,
        compute_qjl=compute_qjl,
    )

    # Graph-safe scatter: clamp negative slots to 0 (padding tokens
    # write harmlessly to block 0, position 0).
    safe_slots = slot_mapping.clamp(min=0)
    block_idx = safe_slots // block_size
    block_offset = safe_slots % block_size

    if num_bits == 4:
        # TQ4: nibble packing -- 2 indices per byte, first/second half split
        head_dim = indices.shape[-1]
        first_half = indices[..., : head_dim // 2]
        second_half = indices[..., head_dim // 2 :]
        packed = ((first_half << 4) | second_half).to(torch.uint8)
        key_cache[block_idx, block_offset] = packed
    else:
        # TQ3 and others: unpacked uint8, 1 byte per index
        key_cache[block_idx, block_offset] = indices

    norm_cache[block_idx, block_offset] = norms

    # Store QJL sign bits and residual norms if computed
    if compute_qjl and qjl_signs is not None:
        qjl_sign_cache[block_idx, block_offset] = qjl_signs
        qjl_rnorm_cache[block_idx, block_offset] = qjl_rnorms


def dequantize_cache_blocks(
    key_cache: torch.Tensor,
    norm_cache: torch.Tensor,
    rotation_matrix: torch.Tensor,
    centroids: torch.Tensor,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize entire key cache blocks for attention computation.

    Args:
        key_cache: [num_blocks, block_size, num_kv_heads, head_dim] uint8
        norm_cache: [num_blocks, block_size, num_kv_heads] float16
        rotation_matrix: [head_dim, head_dim] float32
        centroids: [2^b] float32
        target_dtype: output dtype

    Returns:
        keys: [num_blocks, block_size, num_kv_heads, head_dim] target_dtype
    """
    shape = key_cache.shape  # [B, S, H, D]
    flat_indices = key_cache.reshape(-1, shape[-1])  # [B*S*H, D]
    flat_norms = norm_cache.reshape(-1)  # [B*S*H]

    # Lookup centroids
    rotated_approx = centroids[flat_indices.long()]  # [B*S*H, D] float32

    # Unrotate
    keys_approx = torch.matmul(rotated_approx, rotation_matrix)  # [B*S*H, D]

    # Scale by norms
    keys_approx = keys_approx * flat_norms.float().unsqueeze(-1)

    return keys_approx.reshape(shape).to(target_dtype)


# ---------------------------------------------------------------------------
# Channel outlier separation (automatic for tq3, opt-in via tq4o)
# ---------------------------------------------------------------------------

# Per-layer outlier channel indices (detected at first real batch).
_OUTLIER_CACHE: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
_OUTLIER_COUNTER: int = 0


def get_next_outlier_layer_id() -> int:
    """Get a unique layer ID for outlier detection."""
    global _OUTLIER_COUNTER
    lid = _OUTLIER_COUNTER
    _OUTLIER_COUNTER += 1
    return lid


def detect_outlier_channels(
    keys: torch.Tensor,
    num_outlier: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Detect outlier channels by per-channel average magnitude."""
    chan_mag = keys.float().abs().mean(dim=(0, 1))
    _, sorted_idx = chan_mag.sort(descending=True)
    outlier = sorted_idx[:num_outlier].sort().values
    normal = sorted_idx[num_outlier:].sort().values
    return outlier.to(torch.int64), normal.to(torch.int64)


def get_or_detect_outlier_indices(
    layer_id: int,
    keys: torch.Tensor,
    num_outlier: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get cached outlier indices or detect from keys."""
    if layer_id in _OUTLIER_CACHE:
        return _OUTLIER_CACHE[layer_id]
    outlier, normal = detect_outlier_channels(keys, num_outlier)
    _OUTLIER_CACHE[layer_id] = (outlier, normal)
    return outlier, normal


def quantize_and_store_outlier(
    keys: torch.Tensor,
    outlier_cache: torch.Tensor,
    normal_cache: torch.Tensor,
    normal_norm_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rotation_matrix: torch.Tensor,
    boundaries: torch.Tensor,
    block_size: int,
    outlier_indices: torch.Tensor,
    normal_indices: torch.Tensor,
    num_bits: int = 4,
    centroids: torch.Tensor | None = None,
) -> None:
    """Store outlier channels raw, quantize normal channels with TQ."""
    key_outlier = keys[:, :, outlier_indices]
    key_normal = keys[:, :, normal_indices]

    # Normal: normalize, rotate, quantize
    normal_norms = key_normal.float().norm(dim=-1)
    safe_norms = normal_norms.clamp(min=1e-8)
    key_normal_unit = key_normal.float() / safe_norms.unsqueeze(-1)
    rotated = torch.matmul(key_normal_unit, rotation_matrix.t())
    indices = (
        torch.searchsorted(boundaries, rotated.reshape(-1))
        .reshape(rotated.shape)
        .to(torch.uint8)
    )

    # Scatter
    safe_slots = slot_mapping.clamp(min=0)
    block_idx = safe_slots // block_size
    block_offset = safe_slots % block_size

    # Store outlier raw
    outlier_cache[block_idx, block_offset] = key_outlier.to(outlier_cache.dtype)
    # Store normal norms
    normal_norm_cache[block_idx, block_offset] = normal_norms.to(torch.float16)
    # Pack and store normal indices (pad to cache dim)
    num_normal = indices.shape[-1]
    cache_dim = normal_cache.shape[-1]
    if num_normal % 2 == 0:
        first = indices[..., : num_normal // 2]
        second = indices[..., num_normal // 2 :]
        packed = ((first << 4) | second).to(torch.uint8)
        if packed.shape[-1] < cache_dim:
            pad = torch.zeros(
                *packed.shape[:-1],
                cache_dim - packed.shape[-1],
                dtype=torch.uint8,
                device=packed.device,
            )
            packed = torch.cat([packed, pad], dim=-1)
        normal_cache[block_idx, block_offset] = packed
    else:
        normal_cache[block_idx, block_offset] = indices
