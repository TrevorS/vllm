# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Authors:
#  - Burkhard Ringlein <ngl@zurich.ibm.com>
#  - Jan van Lunteren <jvl@zurich.ibm.com>
#  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
#  - Thomas Parnell <tpa@zurich.ibm.com>

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)
is_batch_invariant = envs.VLLM_BATCH_INVARIANT
float8_info = torch.finfo(current_platform.fp8_dtype())


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.exp(Sdiv)
    p2 = tl.exp(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@triton.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1


@triton.jit
def kernel_unified_attention_2d(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    out_scale,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_ALIBI_SQRT: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    USE_MM_PREFIX: tl.constexpr,  # bool
    MAX_MM_RANGES: tl.constexpr,  # int
    mm_prefix_range_ptr,  # [num_seqs] - prefix length for each sequence
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    # TurboQuant parameters
    TQ_KEYS: tl.constexpr = False,
    tq_norm_cache_ptr=None,
    tq_centroids_ptr=None,
    tq_rotation_ptr=None,
    stride_norm_cache_0: tl.int64 = 0,
    stride_norm_cache_1: tl.int64 = 0,
    stride_norm_cache_2: tl.int64 = 0,
    TQ_NUM_CENTROIDS: tl.constexpr = 16,
    TQ_PACKED: tl.constexpr = False,
    TQ_QJL: tl.constexpr = False,
    tq_qjl_signs_ptr=None,
    tq_qjl_rnorms_ptr=None,
    stride_qjl_signs_0: tl.int64 = 0,
    stride_qjl_signs_1: tl.int64 = 0,
    stride_qjl_signs_2: tl.int64 = 0,
    stride_qjl_signs_3: tl.constexpr = 1,
    stride_qjl_rnorms_0: tl.int64 = 0,
    stride_qjl_rnorms_1: tl.int64 = 0,
    stride_qjl_rnorms_2: tl.int64 = 0,
    TQ_OUTLIER: tl.constexpr = False,
    tq_outlier_ptr=None,
    stride_outlier_0: tl.int64 = 0,
    stride_outlier_1: tl.int64 = 0,
    stride_outlier_2: tl.int64 = 0,
    stride_outlier_3: tl.constexpr = 1,
    TQ_NUM_OUTLIER: tl.constexpr = 0,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    # TurboQuant: Q rotation or outlier split
    if TQ_KEYS:
        TQ_HALF_D: tl.constexpr = HEAD_SIZE_PADDED // 2
        rot_offs_row = tl.arange(0, HEAD_SIZE_PADDED)
        if TQ_OUTLIER:
            # Q is pre-processed in Python as [Q_outlier | Q_normal_rot]
            # and passed via query_ptr. Re-load split parts directly.
            # Note: TQ_NUM_OUTLIER must be power of 2 (e.g., 32)
            TQ_NORMAL_D: tl.constexpr = HEAD_SIZE_PADDED - TQ_NUM_OUTLIER
            TQ_HALF_NORMAL: tl.constexpr = TQ_NORMAL_D // 2
            q_base = query_offset_0[:, None] * query_stride_0 + query_offset_1[:, None] * query_stride_1
            q_mask = query_mask_0[:, None] & query_mask_1[:, None]
            # Outlier: first TQ_NUM_OUTLIER dims of pre-processed Q
            outlier_d = tl.arange(0, TQ_NUM_OUTLIER)
            outlier_q_offset = (
                query_offset_0[:, None] * query_stride_0
                + query_offset_1[:, None] * query_stride_1
                + outlier_d[None, :]
            )
            outlier_q_mask = (
                query_mask_0[:, None] & query_mask_1[:, None]
                & (outlier_d[None, :] < TQ_NUM_OUTLIER)
            )
            Q_outlier_part = tl.load(
                query_ptr + outlier_q_offset,
                mask=outlier_q_mask, other=0.0,
            )
            # Normal first half
            nf_d = tl.arange(0, TQ_HALF_D) + TQ_NUM_OUTLIER
            nf_mask = q_mask & (nf_d[None, :] < TQ_NUM_OUTLIER + TQ_HALF_NORMAL)
            Q_first = tl.load(
                query_ptr + q_base + nf_d[None, :],
                mask=nf_mask, other=0.0,
            )
            # Normal second half
            ns_d = tl.arange(0, TQ_HALF_D) + TQ_NUM_OUTLIER + TQ_HALF_NORMAL
            ns_mask = q_mask & (ns_d[None, :] < HEAD_SIZE)
            Q_second = tl.load(
                query_ptr + q_base + ns_d[None, :],
                mask=ns_mask, other=0.0,
            )
        elif TQ_PACKED or TQ_QJL:
            pass  # Q_outlier_part initialized above
            rot_col_first = tl.arange(0, TQ_HALF_D)
            rot_col_second = tl.arange(0, TQ_HALF_D) + TQ_HALF_D
            # Load R^T directly (pre-transposed, row-major)
            R_first = tl.load(
                tq_rotation_ptr
                + rot_offs_row[:, None] * HEAD_SIZE
                + rot_col_first[None, :],
                mask=(rot_offs_row[:, None] < HEAD_SIZE)
                & (rot_col_first[None, :] < HEAD_SIZE),
                other=0.0,
            )
            R_second = tl.load(
                tq_rotation_ptr
                + rot_offs_row[:, None] * HEAD_SIZE
                + rot_col_second[None, :],
                mask=(rot_offs_row[:, None] < HEAD_SIZE)
                & (rot_col_second[None, :] < HEAD_SIZE),
                other=0.0,
            )
            Q_first = tl.dot(Q.to(tl.float32), R_first).to(Q.dtype)
            Q_second = tl.dot(Q.to(tl.float32), R_second).to(Q.dtype)
        else:
            # Load R^T directly (pre-transposed, row-major)
            rot_offs_col = tl.arange(0, HEAD_SIZE_PADDED)
            R_T = tl.load(
                tq_rotation_ptr
                + rot_offs_row[:, None] * HEAD_SIZE
                + rot_offs_col[None, :],
                mask=(rot_offs_row[:, None] < HEAD_SIZE)
                & (rot_offs_col[None, :] < HEAD_SIZE),
                other=0.0,
            )
            Q = tl.dot(Q.to(tl.float32), R_T).to(Q.dtype)
            pass  # Q_outlier_part initialized above

    block_table_offset = seq_idx * block_table_stride

    if not USE_SINKS:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        M = tl.load(
            sink_ptr + query_offset_1,
            mask=query_mask_1,
            other=float("-inf"),
        ).to(dtype=tl.float32)

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)
    if TQ_NUM_OUTLIER > 0:
        Q_outlier_part = tl.zeros([BLOCK_M, TQ_NUM_OUTLIER], dtype=Q.dtype)
    else:
        Q_outlier_part = tl.zeros([BLOCK_M, 1], dtype=Q.dtype)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )  # shape: [BLOCK_M]

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )

    if USE_MM_PREFIX:
        # image bidirectional attention ranges require a full range
        # including q_block padding to make sure doc mask is correct
        max_seq_prefix_len = tl.maximum(max_seq_prefix_len, seq_len)
    else:
        # adjust for potential padding in the last q_block by considering the
        # actual sequence length
        max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # ---- Sliding-window tile pruning --------------------
    # Default: keep previous global behavior
    tile_start = 0
    tile_end = num_tiles
    # TODO(Isotr0py): sliding window pruning with image bidirectional mask
    if SLIDING_WINDOW > 0 and not USE_MM_PREFIX:
        # Query rows covered by this Q-block
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        # For sliding window, each query position q can only attend to
        # keys in the range [q_abs - SLIDING_WINDOW + 1, q_abs]
        # where q_abs = context_len + q
        # The union of allowed key positions for this Q-block is:
        # [context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        # Convert to tile indices and clamp
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    # iterate through tiles (now limited to the sliding window range)
    for j in range(tile_start, tile_end):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )

        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )

        # K : (HEAD_SIZE, TILE_SIZE) or TQ centroid gather
        if TQ_KEYS:
            if TQ_PACKED:
                # Nibble-packed TQ4: head_size//2 uint8 per position
                offs_d_packed = tl.arange(0, TQ_HALF_D)
                packed_dim_mask = tl.where(
                    offs_d_packed < HEAD_SIZE // 2, 1, 0
                ).to(tl.int1)
                k_packed_offset = (
                    physical_block_idx[None, :] * stride_k_cache_0
                    + kv_head_idx * stride_k_cache_2
                    + offs_d_packed[:, None] * stride_k_cache_3
                    + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
                )
                K_packed = tl.load(
                    key_cache_ptr + k_packed_offset,
                    mask=packed_dim_mask[:, None] & tile_mask[None, :],
                    other=0,
                ).to(tl.int32)
                K_hi = tl.load(
                    tq_centroids_ptr + ((K_packed >> 4) & 0xF)
                ).to(Q.dtype)
                K_lo = tl.load(
                    tq_centroids_ptr + (K_packed & 0xF)
                ).to(Q.dtype)
            else:
                # Unpacked: direct uint8 index → centroid
                K_idx = tl.load(
                    key_cache_ptr + k_offset,
                    mask=dim_mask[:, None] & tile_mask[None, :],
                    other=0,
                )
                K = tl.load(
                    tq_centroids_ptr + K_idx.to(tl.int32)
                ).to(Q.dtype)
            # Per-position key norms: [TILE_SIZE]
            norm_offset = (
                physical_block_idx * stride_norm_cache_0
                + kv_head_idx * stride_norm_cache_2
                + (seq_offset % BLOCK_SIZE) * stride_norm_cache_1
            )
            K_norms = tl.load(
                tq_norm_cache_ptr + norm_offset,
                mask=tile_mask,
                other=0.0,
            ).to(tl.float32)
        else:
            K_load = tl.load(
                key_cache_ptr + k_offset,
                mask=dim_mask[:, None] & tile_mask[None, :],
                other=0.0,
            )
            if K_load.dtype.is_fp8():
                if Q.dtype.is_fp8():
                    K = K_load
                else:
                    K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
            else:
                K = K_load.to(Q.dtype)

        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & tile_mask[:, None],
            other=0.0,
        )

        if V_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        # Compute attention mask: causal by default (key <= query)
        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = seq_offset[None, :] <= query_abs_pos

        # Apply sliding window to base mask BEFORE mm_prefix OR.
        # Order must match FlexAttention: (causal AND sliding_window) OR mm_prefix
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)

        # PrefixLM: extend mask with bidirectional ranges for multimodal tokens.
        # Applied AFTER sliding window so mm_prefix ranges override SW restriction.
        if USE_MM_PREFIX:
            for i in range(MAX_MM_RANGES):
                range_start = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2
                )
                range_end = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1
                )

                is_valid = range_start < range_end
                q_in_range = (
                    (query_abs_pos >= range_start)
                    & (query_abs_pos <= range_end)
                    & is_valid
                )
                k_in_range = (
                    (seq_offset[None, :] >= range_start)
                    & (seq_offset[None, :] <= range_end)
                    & is_valid
                )
                seq_mask |= q_in_range & k_in_range

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)

        if TQ_KEYS:
            if TQ_OUTLIER:
                # Load outlier channels (FP16) from outlier cache
                offs_outlier = tl.arange(0, TQ_NUM_OUTLIER)
                outlier_offset = (
                    physical_block_idx[None, :] * stride_outlier_0
                    + kv_head_idx * stride_outlier_2
                    + offs_outlier[:, None] * stride_outlier_3
                    + (seq_offset % BLOCK_SIZE)[None, :]
                    * stride_outlier_1
                )
                outlier_mask = (
                    (offs_outlier[:, None] < TQ_NUM_OUTLIER)
                    & tile_mask[None, :]
                )
                K_outlier = tl.load(
                    tq_outlier_ptr + outlier_offset,
                    mask=outlier_mask,
                    other=0.0,
                ).to(Q.dtype)

                # Outlier dot (raw) + normal dot (TQ) * normal_norm
                outlier_dot = tl.dot(Q_outlier_part, K_outlier)
                if TQ_PACKED:
                    normal_dot = (
                        tl.dot(Q_first, K_hi)
                        + tl.dot(Q_second, K_lo)
                    )
                else:
                    normal_dot = tl.dot(Q, K)  # unpacked outlier not yet impl
                S += scale * (
                    outlier_dot + normal_dot * K_norms[None, :]
                )
            elif TQ_PACKED:
                # Split-dot: first-half @ hi + second-half @ lo
                S += scale * (
                    tl.dot(Q_first, K_hi) + tl.dot(Q_second, K_lo)
                ) * K_norms[None, :]
            else:
                S += scale * tl.dot(Q, K) * K_norms[None, :]

            # QJL residual correction: unbiased inner product estimate
            if TQ_QJL:
                TQ_HALF_SIGN_BYTES: tl.constexpr = HEAD_SIZE_PADDED // 16
                d_half = tl.arange(0, HEAD_SIZE_PADDED // 2)
                byte_idx_h = d_half // 8
                bit_pos_h = 7 - (d_half % 8)
                dim_mask_h = tl.where(
                    d_half < HEAD_SIZE // 2, 1, 0
                ).to(tl.int1)

                qjl_base = (
                    physical_block_idx[None, :] * stride_qjl_signs_0
                    + kv_head_idx * stride_qjl_signs_2
                    + (seq_offset % BLOCK_SIZE)[None, :]
                    * stride_qjl_signs_1
                )
                # First half signs
                sb_first = tl.load(
                    tq_qjl_signs_ptr + qjl_base
                    + byte_idx_h[:, None] * stride_qjl_signs_3,
                    mask=dim_mask_h[:, None] & tile_mask[None, :],
                    other=0,
                ).to(tl.int32)
                signs_first = (
                    ((sb_first >> bit_pos_h[:, None]) & 1) * 2 - 1
                ).to(tl.float32)
                # Second half signs
                sb_second = tl.load(
                    tq_qjl_signs_ptr + qjl_base
                    + (byte_idx_h[:, None] + TQ_HALF_SIGN_BYTES)
                    * stride_qjl_signs_3,
                    mask=dim_mask_h[:, None] & tile_mask[None, :],
                    other=0,
                ).to(tl.int32)
                signs_second = (
                    ((sb_second >> bit_pos_h[:, None]) & 1) * 2 - 1
                ).to(tl.float32)

                # Residual norms: [TILE_SIZE]
                rnorm_offset = (
                    physical_block_idx * stride_qjl_rnorms_0
                    + kv_head_idx * stride_qjl_rnorms_2
                    + (seq_offset % BLOCK_SIZE) * stride_qjl_rnorms_1
                )
                residual_norms = tl.load(
                    tq_qjl_rnorms_ptr + rnorm_offset,
                    mask=tile_mask,
                    other=0.0,
                ).to(tl.float32)

                # sqrt(pi/2)/D * dot(Q_rot, signs) * r_norm * k_norm
                TQ_QJL_CONST: tl.constexpr = 1.2533141373
                qjl_dot = (
                    tl.dot(Q_first.to(tl.float32), signs_first)
                    + tl.dot(Q_second.to(tl.float32), signs_second)
                )
                S += (
                    scale
                    * (TQ_QJL_CONST / HEAD_SIZE)
                    * qjl_dot
                    * residual_norms[None, :]
                    * K_norms[None, :]
                )
        else:
            S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            if USE_ALIBI_SQRT:
                relative_pos = seq_offset - (context_len + query_pos[:, None])
                alibi_offset = tl.where(
                    relative_pos <= 0,
                    -tl.sqrt((-relative_pos).to(tl.float32)),
                    0.0,
                )
            else:
                alibi_offset = seq_offset - context_len
            S += alibi_slope[:, None] * alibi_offset

        if USE_QQ_BIAS:
            # compute key positions relative to query section
            key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
            # load bias only for keys that correspond to queries
            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],  # avoid OOB for context keys
                other=0.0,
            )
            S += qq_bias

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE)
        P = tl.exp(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.exp(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW, V, 0.0
            )

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc += tl.dot(P.to(V.dtype), V)

    # epilogue
    acc = acc / L[:, None]
    if USE_FP8:
        acc = acc * tl.load(out_scale)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


@triton.jit
def kernel_unified_attention_3d(
    segm_output_ptr,
    # [num_tokens, num_query_heads, num_segments, head_size_padded]
    segm_max_ptr,  # [num_tokens, num_query_heads, num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, num_segments]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, num_kv_heads, head_size // x, blk_size, x]
    value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_ALIBI_SQRT: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    USE_MM_PREFIX: tl.constexpr,  # bool
    MAX_MM_RANGES: tl.constexpr,  # int
    mm_prefix_range_ptr,  # [num_seqs] - prefix length for each sequence
    # TurboQuant parameters (3D kernel: unpacked only, no split-dot/QJL)
    TQ_KEYS: tl.constexpr = False,
    tq_norm_cache_ptr=None,
    tq_centroids_ptr=None,
    tq_rotation_ptr=None,
    stride_norm_cache_0: tl.int64 = 0,
    stride_norm_cache_1: tl.int64 = 0,
    stride_norm_cache_2: tl.int64 = 0,
    TQ_NUM_CENTROIDS: tl.constexpr = 16,
    TQ_PACKED: tl.constexpr = False,
    TQ_QJL: tl.constexpr = False,
    tq_qjl_signs_ptr=None,
    tq_qjl_rnorms_ptr=None,
    stride_qjl_signs_0: tl.int64 = 0,
    stride_qjl_signs_1: tl.int64 = 0,
    stride_qjl_signs_2: tl.int64 = 0,
    stride_qjl_signs_3: tl.constexpr = 1,
    stride_qjl_rnorms_0: tl.int64 = 0,
    stride_qjl_rnorms_1: tl.int64 = 0,
    stride_qjl_rnorms_2: tl.int64 = 0,
    TQ_OUTLIER: tl.constexpr = False,
    tq_outlier_ptr=None,
    stride_outlier_0: tl.int64 = 0,
    stride_outlier_1: tl.int64 = 0,
    stride_outlier_2: tl.int64 = 0,
    stride_outlier_3: tl.constexpr = 1,
    TQ_NUM_OUTLIER: tl.constexpr = 0,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    # TurboQuant: pre-rotate Q (3D kernel)
    if TQ_KEYS:
        TQ_HALF_D: tl.constexpr = HEAD_SIZE_PADDED // 2
        rot_offs_row = tl.arange(0, HEAD_SIZE_PADDED)
        if TQ_PACKED or TQ_QJL:
            rot_col_first = tl.arange(0, TQ_HALF_D)
            rot_col_second = tl.arange(0, TQ_HALF_D) + TQ_HALF_D
            # Load R^T directly (pre-transposed, row-major)
            R_first = tl.load(
                tq_rotation_ptr
                + rot_offs_row[:, None] * HEAD_SIZE
                + rot_col_first[None, :],
                mask=(rot_offs_row[:, None] < HEAD_SIZE)
                & (rot_col_first[None, :] < HEAD_SIZE),
                other=0.0,
            )
            R_second = tl.load(
                tq_rotation_ptr
                + rot_offs_row[:, None] * HEAD_SIZE
                + rot_col_second[None, :],
                mask=(rot_offs_row[:, None] < HEAD_SIZE)
                & (rot_col_second[None, :] < HEAD_SIZE),
                other=0.0,
            )
            Q_first = tl.dot(Q.to(tl.float32), R_first).to(Q.dtype)
            Q_second = tl.dot(Q.to(tl.float32), R_second).to(Q.dtype)
        else:
            # Load R^T directly (pre-transposed, row-major)
            rot_offs_col = tl.arange(0, HEAD_SIZE_PADDED)
            R_T = tl.load(
                tq_rotation_ptr
                + rot_offs_row[:, None] * HEAD_SIZE
                + rot_offs_col[None, :],
                mask=(rot_offs_row[:, None] < HEAD_SIZE)
                & (rot_offs_col[None, :] < HEAD_SIZE),
                other=0.0,
            )
            Q = tl.dot(Q.to(tl.float32), R_T).to(Q.dtype)
            pass  # Q_outlier_part initialized above

    block_table_offset = seq_idx * block_table_stride

    if USE_SINKS:
        if segm_idx == 0:
            M = tl.load(
                sink_ptr + query_offset_1,
                mask=query_mask_1,
                other=float("-inf"),
            ).to(dtype=tl.float32)
        else:
            M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)
    if TQ_NUM_OUTLIER > 0:
        Q_outlier_part = tl.zeros([BLOCK_M, TQ_NUM_OUTLIER], dtype=Q.dtype)
    else:
        Q_outlier_part = tl.zeros([BLOCK_M, 1], dtype=Q.dtype)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )  # shape: [BLOCK_M]

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )

    # adjust for potential padding in the last q_block by considering the
    # actual sequence length
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # ---- Sliding-window tile pruning --------------------
    # Default: keep previous global behavior
    tile_start = 0
    tile_end = num_tiles
    # TODO(Isotr0py): sliding window pruning with image bidirectional mask
    if SLIDING_WINDOW > 0 and not USE_MM_PREFIX:
        # Query rows covered by this Q-block
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        # For sliding window, each query position q can only attend to
        # keys in the range [q_abs - SLIDING_WINDOW + 1, q_abs]
        # where q_abs = context_len + q
        # The union of allowed key positions for this Q-block is:
        # [context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        # Convert to tile indices and clamp
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    # iterate through tiles (now limited to the sliding window range)
    for j in range(
        max(segm_idx * tiles_per_segment, tile_start),
        min((segm_idx + 1) * tiles_per_segment, tile_end),
    ):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )

        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )

        # K : (HEAD_SIZE, TILE_SIZE) or TQ centroid gather
        if TQ_KEYS:
            if TQ_PACKED:
                # Nibble-packed TQ4: head_size//2 uint8 per position
                offs_d_packed = tl.arange(0, TQ_HALF_D)
                packed_dim_mask = tl.where(
                    offs_d_packed < HEAD_SIZE // 2, 1, 0
                ).to(tl.int1)
                k_packed_offset = (
                    physical_block_idx[None, :] * stride_k_cache_0
                    + kv_head_idx * stride_k_cache_2
                    + offs_d_packed[:, None] * stride_k_cache_3
                    + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
                )
                K_packed = tl.load(
                    key_cache_ptr + k_packed_offset,
                    mask=packed_dim_mask[:, None] & tile_mask[None, :],
                    other=0,
                ).to(tl.int32)
                K_hi = tl.load(
                    tq_centroids_ptr + ((K_packed >> 4) & 0xF)
                ).to(Q.dtype)
                K_lo = tl.load(
                    tq_centroids_ptr + (K_packed & 0xF)
                ).to(Q.dtype)
            else:
                # Unpacked: direct uint8 index → centroid
                K_idx = tl.load(
                    key_cache_ptr + k_offset,
                    mask=dim_mask[:, None] & tile_mask[None, :],
                    other=0,
                )
                K = tl.load(
                    tq_centroids_ptr + K_idx.to(tl.int32)
                ).to(Q.dtype)
            # Per-position key norms: [TILE_SIZE]
            norm_offset = (
                physical_block_idx * stride_norm_cache_0
                + kv_head_idx * stride_norm_cache_2
                + (seq_offset % BLOCK_SIZE) * stride_norm_cache_1
            )
            K_norms = tl.load(
                tq_norm_cache_ptr + norm_offset,
                mask=tile_mask,
                other=0.0,
            ).to(tl.float32)
        else:
            K_load = tl.load(
                key_cache_ptr + k_offset,
                mask=dim_mask[:, None] & tile_mask[None, :],
                other=0.0,
            )
            if K_load.dtype.is_fp8():
                if Q.dtype.is_fp8():
                    K = K_load
                else:
                    K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
            else:
                K = K_load.to(Q.dtype)

        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & tile_mask[:, None],
            other=0.0,
        )

        if V_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        # Compute attention mask: causal by default (key <= query)
        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = seq_offset[None, :] <= query_abs_pos

        # Apply sliding window to base mask BEFORE mm_prefix OR.
        # Order must match FlexAttention: (causal AND sliding_window) OR mm_prefix
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)

        # PrefixLM: extend mask with bidirectional ranges for multimodal tokens.
        # Applied AFTER sliding window so mm_prefix ranges override SW restriction.
        if USE_MM_PREFIX:
            for i in range(MAX_MM_RANGES):
                range_start = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2
                )
                range_end = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1
                )

                is_valid = range_start < range_end
                q_in_range = (
                    (query_abs_pos >= range_start)
                    & (query_abs_pos <= range_end)
                    & is_valid
                )
                k_in_range = (
                    (seq_offset[None, :] >= range_start)
                    & (seq_offset[None, :] <= range_end)
                    & is_valid
                )
                seq_mask |= q_in_range & k_in_range

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        if TQ_KEYS:
            if TQ_OUTLIER:
                # Load outlier channels (FP16) from outlier cache
                offs_outlier = tl.arange(0, TQ_NUM_OUTLIER)
                outlier_offset = (
                    physical_block_idx[None, :] * stride_outlier_0
                    + kv_head_idx * stride_outlier_2
                    + offs_outlier[:, None] * stride_outlier_3
                    + (seq_offset % BLOCK_SIZE)[None, :]
                    * stride_outlier_1
                )
                outlier_mask = (
                    (offs_outlier[:, None] < TQ_NUM_OUTLIER)
                    & tile_mask[None, :]
                )
                K_outlier = tl.load(
                    tq_outlier_ptr + outlier_offset,
                    mask=outlier_mask,
                    other=0.0,
                ).to(Q.dtype)

                # Outlier dot (raw) + normal dot (TQ) * normal_norm
                outlier_dot = tl.dot(Q_outlier_part, K_outlier)
                if TQ_PACKED:
                    normal_dot = (
                        tl.dot(Q_first, K_hi)
                        + tl.dot(Q_second, K_lo)
                    )
                else:
                    normal_dot = tl.dot(Q, K)  # unpacked outlier not yet impl
                S += scale * (
                    outlier_dot + normal_dot * K_norms[None, :]
                )
            elif TQ_PACKED:
                # Split-dot: first-half @ hi + second-half @ lo
                S += scale * (
                    tl.dot(Q_first, K_hi) + tl.dot(Q_second, K_lo)
                ) * K_norms[None, :]
            else:
                S += scale * tl.dot(Q, K) * K_norms[None, :]

            # QJL residual correction: unbiased inner product estimate
            if TQ_QJL:
                TQ_HALF_SIGN_BYTES: tl.constexpr = HEAD_SIZE_PADDED // 16
                d_half = tl.arange(0, HEAD_SIZE_PADDED // 2)
                byte_idx_h = d_half // 8
                bit_pos_h = 7 - (d_half % 8)
                dim_mask_h = tl.where(
                    d_half < HEAD_SIZE // 2, 1, 0
                ).to(tl.int1)

                qjl_base = (
                    physical_block_idx[None, :] * stride_qjl_signs_0
                    + kv_head_idx * stride_qjl_signs_2
                    + (seq_offset % BLOCK_SIZE)[None, :]
                    * stride_qjl_signs_1
                )
                # First half signs
                sb_first = tl.load(
                    tq_qjl_signs_ptr + qjl_base
                    + byte_idx_h[:, None] * stride_qjl_signs_3,
                    mask=dim_mask_h[:, None] & tile_mask[None, :],
                    other=0,
                ).to(tl.int32)
                signs_first = (
                    ((sb_first >> bit_pos_h[:, None]) & 1) * 2 - 1
                ).to(tl.float32)
                # Second half signs
                sb_second = tl.load(
                    tq_qjl_signs_ptr + qjl_base
                    + (byte_idx_h[:, None] + TQ_HALF_SIGN_BYTES)
                    * stride_qjl_signs_3,
                    mask=dim_mask_h[:, None] & tile_mask[None, :],
                    other=0,
                ).to(tl.int32)
                signs_second = (
                    ((sb_second >> bit_pos_h[:, None]) & 1) * 2 - 1
                ).to(tl.float32)

                # Residual norms: [TILE_SIZE]
                rnorm_offset = (
                    physical_block_idx * stride_qjl_rnorms_0
                    + kv_head_idx * stride_qjl_rnorms_2
                    + (seq_offset % BLOCK_SIZE) * stride_qjl_rnorms_1
                )
                residual_norms = tl.load(
                    tq_qjl_rnorms_ptr + rnorm_offset,
                    mask=tile_mask,
                    other=0.0,
                ).to(tl.float32)

                # sqrt(pi/2)/D * dot(Q_rot, signs) * r_norm * k_norm
                TQ_QJL_CONST: tl.constexpr = 1.2533141373
                qjl_dot = (
                    tl.dot(Q_first.to(tl.float32), signs_first)
                    + tl.dot(Q_second.to(tl.float32), signs_second)
                )
                S += (
                    scale
                    * (TQ_QJL_CONST / HEAD_SIZE)
                    * qjl_dot
                    * residual_norms[None, :]
                    * K_norms[None, :]
                )
        else:
            S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            if USE_ALIBI_SQRT:
                relative_pos = seq_offset - (context_len + query_pos[:, None])
                alibi_offset = tl.where(
                    relative_pos <= 0,
                    -tl.sqrt((-relative_pos).to(tl.float32)),
                    0.0,
                )
            else:
                alibi_offset = seq_offset - context_len
            S += alibi_slope[:, None] * alibi_offset

        if USE_QQ_BIAS:
            # compute key positions relative to query section
            key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
            # load bias only for keys that correspond to queries
            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],  # avoid OOB for context keys
                other=0.0,
            )
            S += qq_bias

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE,)
        P = tl.exp(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.exp(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW, V, 0.0
            )

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc += tl.dot(P.to(V.dtype), V)

    segm_output_offset = (
        query_offset_0[:, None].to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )
    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


@triton.jit
def reduce_segments(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    out_scale_inv,  # float32
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    block_table_stride: tl.int64,  # int
    TILE_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # load segment maxima
    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # write result
    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


def _is_gemma3_attention(head_size: int, sliding_window: int) -> bool:
    """Detect Gemma3 models via unique (head_size, sliding_window) signature.

    Gemma3 models are the only ones using sliding_window=1024 with
    head_size 128 (27B) or 256 (1B, 4B, 12B). Other SWA models use
    different window sizes (Mistral=4096, Phi-3=2047).
    """
    return sliding_window == 1024 and head_size in (128, 256)


def _get_tile_size(
    head_size: int,
    sliding_window: int,
    element_size: int,
    is_prefill: bool,
) -> int:
    """Select tile size with Gemma3-specific optimization.

    For Gemma3, use 32 for both prefill and decode to better utilize
    the larger head dimension (128/256). For other models, use
    the default vLLM behavior.
    """
    if _is_gemma3_attention(head_size, sliding_window):
        # Gemma3: use 32 for decode (default is 16)
        return 32

    # Default behavior
    if is_prefill:
        return 32
    return 16 if element_size >= 2 else 32


def unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    seq_threshold_3D=None,
    num_par_softmax_segments=None,
    softmax_segm_output=None,
    softmax_segm_max=None,
    softmax_segm_expsum=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    # Optional tensor for sinks
    sinks=None,
    # Optional tensor for prefix lengths (PrefixLM support)
    mm_prefix_range=None,
    use_alibi_sqrt=False,
    # TurboQuant parameters
    tq_norm_cache=None,
    tq_centroids=None,
    tq_rotation_matrix=None,
    tq_num_centroids=16,
    tq_packed=False,
    tq_qjl_signs=None,
    tq_qjl_rnorms=None,
    tq_outlier_cache=None,
    tq_num_outlier=0,
):
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    use_mm_prefix = False
    max_mm_ranges = 0
    if mm_prefix_range is not None:
        if mm_prefix_range.ndim == 3:
            use_mm_prefix = True
            max_mm_ranges = mm_prefix_range.shape[1]
        else:
            raise ValueError(
                f"Unsupported mm_prefix_range shape: {mm_prefix_range.shape}"
            )

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    # Tile sizes for prefill and decode. Gemma3 models use optimized values.
    # Note: tile size must be at least 32 for fp8 (element_size == 1).
    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0
    TILE_SIZE_PREFILL = _get_tile_size(
        head_size,
        sliding_window_val,
        q.element_size(),
        is_prefill=True,
    )
    TILE_SIZE_DECODE = _get_tile_size(
        head_size,
        sliding_window_val,
        q.element_size(),
        is_prefill=False,
    )

    # Launch the 2D kernel if
    # 1. No intermediate tiled softmax buffers for the 3D kernel have been allocated, or
    # 2. The batch includes at least one prefill request, or
    # 3. The number of sequences exceeds the configured threshold, or
    # 4. Batch invariance is enabled
    if (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or max_seqlen_q > 1
        or num_seqs > seq_threshold_3D
        or is_batch_invariant
    ):
        kernel_unified_attention_2d[
            (
                total_num_q_blocks,
                num_kv_heads,
            )
        ](
            output_ptr=out,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            sink_ptr=sinks,
            block_tables_ptr=block_table,
            seq_lens_ptr=seqused_k,
            alibi_slopes_ptr=alibi_slopes,
            qq_bias_ptr=qq_bias,
            scale=softmax_scale,
            k_scale=k_descale,
            v_scale=v_descale,
            out_scale=1 / output_scale if output_scale is not None else 1.0,
            softcap=softcap,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
            BLOCK_SIZE=block_size,
            TILE_SIZE=TILE_SIZE_PREFILL,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            USE_ALIBI_SQRT=use_alibi_sqrt,
            USE_QQ_BIAS=use_qq_bias,
            USE_SOFTCAP=(softcap > 0),
            USE_SINKS=(sinks is not None),
            USE_MM_PREFIX=use_mm_prefix,
            MAX_MM_RANGES=max_mm_ranges,
            mm_prefix_range_ptr=mm_prefix_range,
            SLIDING_WINDOW=(1 + window_size[0]),
            stride_k_cache_0=k.stride(0),
            stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2),
            stride_v_cache_3=v.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            USE_FP8=output_scale is not None,
            # TurboQuant
            TQ_KEYS=tq_norm_cache is not None,
            tq_norm_cache_ptr=tq_norm_cache,
            tq_centroids_ptr=tq_centroids,
            tq_rotation_ptr=tq_rotation_matrix,
            stride_norm_cache_0=tq_norm_cache.stride(0) if tq_norm_cache is not None else 0,
            stride_norm_cache_1=tq_norm_cache.stride(1) if tq_norm_cache is not None else 0,
            stride_norm_cache_2=tq_norm_cache.stride(2) if tq_norm_cache is not None else 0,
            TQ_NUM_CENTROIDS=tq_num_centroids,
            TQ_PACKED=tq_packed,
            TQ_QJL=tq_qjl_signs is not None,
            tq_qjl_signs_ptr=tq_qjl_signs,
            tq_qjl_rnorms_ptr=tq_qjl_rnorms,
            stride_qjl_signs_0=tq_qjl_signs.stride(0) if tq_qjl_signs is not None else 0,
            stride_qjl_signs_1=tq_qjl_signs.stride(1) if tq_qjl_signs is not None else 0,
            stride_qjl_signs_2=tq_qjl_signs.stride(2) if tq_qjl_signs is not None else 0,
            stride_qjl_rnorms_0=tq_qjl_rnorms.stride(0) if tq_qjl_rnorms is not None else 0,
            stride_qjl_rnorms_1=tq_qjl_rnorms.stride(1) if tq_qjl_rnorms is not None else 0,
            stride_qjl_rnorms_2=tq_qjl_rnorms.stride(2) if tq_qjl_rnorms is not None else 0,
            TQ_OUTLIER=tq_outlier_cache is not None,
            tq_outlier_ptr=tq_outlier_cache,
            stride_outlier_0=tq_outlier_cache.stride(0) if tq_outlier_cache is not None else 0,
            stride_outlier_1=tq_outlier_cache.stride(1) if tq_outlier_cache is not None else 0,
            stride_outlier_2=tq_outlier_cache.stride(2) if tq_outlier_cache is not None else 0,
            stride_outlier_3=tq_outlier_cache.stride(3) if tq_outlier_cache is not None else 1,
            TQ_NUM_OUTLIER=tq_num_outlier,
        )
    else:
        kernel_unified_attention_3d[
            (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)
        ](
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            sink_ptr=sinks,
            block_tables_ptr=block_table,
            seq_lens_ptr=seqused_k,
            alibi_slopes_ptr=alibi_slopes,
            qq_bias_ptr=qq_bias,
            scale=softmax_scale,
            k_scale=k_descale,
            v_scale=v_descale,
            softcap=softcap,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
            BLOCK_SIZE=block_size,
            TILE_SIZE=TILE_SIZE_DECODE,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            USE_ALIBI_SQRT=use_alibi_sqrt,
            USE_QQ_BIAS=use_qq_bias,
            USE_SOFTCAP=(softcap > 0),
            USE_SINKS=(sinks is not None),
            USE_MM_PREFIX=use_mm_prefix,
            MAX_MM_RANGES=max_mm_ranges,
            mm_prefix_range_ptr=mm_prefix_range,
            SLIDING_WINDOW=(1 + window_size[0]),
            stride_k_cache_0=k.stride(0),
            stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2),
            stride_v_cache_3=v.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            # TurboQuant
            TQ_KEYS=tq_norm_cache is not None,
            tq_norm_cache_ptr=tq_norm_cache,
            tq_centroids_ptr=tq_centroids,
            tq_rotation_ptr=tq_rotation_matrix,
            stride_norm_cache_0=tq_norm_cache.stride(0) if tq_norm_cache is not None else 0,
            stride_norm_cache_1=tq_norm_cache.stride(1) if tq_norm_cache is not None else 0,
            stride_norm_cache_2=tq_norm_cache.stride(2) if tq_norm_cache is not None else 0,
            TQ_NUM_CENTROIDS=tq_num_centroids,
            TQ_PACKED=tq_packed,
            TQ_QJL=tq_qjl_signs is not None,
            tq_qjl_signs_ptr=tq_qjl_signs,
            tq_qjl_rnorms_ptr=tq_qjl_rnorms,
            stride_qjl_signs_0=tq_qjl_signs.stride(0) if tq_qjl_signs is not None else 0,
            stride_qjl_signs_1=tq_qjl_signs.stride(1) if tq_qjl_signs is not None else 0,
            stride_qjl_signs_2=tq_qjl_signs.stride(2) if tq_qjl_signs is not None else 0,
            stride_qjl_rnorms_0=tq_qjl_rnorms.stride(0) if tq_qjl_rnorms is not None else 0,
            stride_qjl_rnorms_1=tq_qjl_rnorms.stride(1) if tq_qjl_rnorms is not None else 0,
            stride_qjl_rnorms_2=tq_qjl_rnorms.stride(2) if tq_qjl_rnorms is not None else 0,
            TQ_OUTLIER=tq_outlier_cache is not None,
            tq_outlier_ptr=tq_outlier_cache,
            stride_outlier_0=tq_outlier_cache.stride(0) if tq_outlier_cache is not None else 0,
            stride_outlier_1=tq_outlier_cache.stride(1) if tq_outlier_cache is not None else 0,
            stride_outlier_2=tq_outlier_cache.stride(2) if tq_outlier_cache is not None else 0,
            stride_outlier_3=tq_outlier_cache.stride(3) if tq_outlier_cache is not None else 1,
            TQ_NUM_OUTLIER=tq_num_outlier,
        )
        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            out_scale_inv=1 / output_scale if output_scale is not None else 1.0,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            TILE_SIZE=TILE_SIZE_DECODE,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            USE_FP8=output_scale is not None,
        )
