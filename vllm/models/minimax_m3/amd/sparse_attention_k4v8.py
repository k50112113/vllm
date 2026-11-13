# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3 block-sparse attention over a k4v8 K/V cache on ROCm (gfx950).

k4v8: K in MXFP4 (E8M0 scale per 32 dims, preshuffled for the scaled MFMA),
V in fp8 e4m3 with the layer's per-tensor v_scale. The attend (Gluon or
Triton) and the cache store are AITER's MSA (`aiter.ops.triton.attention.msa`).

Opt in with `VLLM_ROCM_MINIMAX_M3_KV_K4V8=1` on top of `--kv-cache-dtype nvfp4`:
the index cache and the indexer stay NVFP4 exactly as without it, only the
main K/V cache and its attend change.

  VLLM_ROCM_MINIMAX_M3_K4V8_BACKEND gluon (default, gfx950) | triton
  VLLM_ROCM_MINIMAX_M3_K4V8_Q_MODE  mxfp8 (default) | mxfp8x2 | bf16 (triton only)
  VLLM_ROCM_MINIMAX_M3_K4V8_P_BF16  1: bf16 P against the upcast V (default fp8 P)
  VLLM_ROCM_MINIMAX_M3_K4V8_FUSED_SORT_TOPK  0: the prefill top-k without the
      sort_topk epilogue (the Gluon attend then builds its sort_topk itself);
      default 1: AITER's FlyDSL prefill top-k emits it

Prefill sort_topk: the Gluon attend walks 2 query rows per program (BLOCK_M 32)
over their sort_topk, the rows' top-k pages merged, deduplicated and sorted by
page. The indexer's prefill top-k (AITER's
flydsl_fused_m3_index_topk_prefill_indices_sort_topk) writes it as it selects,
from this layer's attend block table and row positions, so the attend skips
AITER's msa_sort_topk_kernel pre-pass.
"""

import os
from dataclasses import replace
from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context
from vllm.models.minimax_m3.amd.sparse_attention_nvfp4 import (
    MiniMaxM3SparseRocmNvfp4Backend,
    rocm_nvfp4_unsupported_reason,
)
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseImpl,
    MiniMaxM3SparseMetadata,
)
from vllm.v1.attention.backend import AttentionLayer
from vllm.v1.kv_cache_interface import AttentionSpec

# Bytes per token per KV head: 64 MXFP4 K + 4 E8M0 K scales + 128 fp8 V.
K4V8_TOKEN_BYTES = 128 // 2 + 128 // 32 + 128


def minimax_m3_k4v8_enabled() -> bool:
    return os.getenv("VLLM_ROCM_MINIMAX_M3_KV_K4V8", "0") in ("1", "true", "True")


def _q_mode() -> str:
    return os.getenv("VLLM_ROCM_MINIMAX_M3_K4V8_Q_MODE", "mxfp8")


def _backend() -> str:
    return os.getenv("VLLM_ROCM_MINIMAX_M3_K4V8_BACKEND", "gluon")


def _p_fp8() -> bool:
    return os.getenv("VLLM_ROCM_MINIMAX_M3_K4V8_P_BF16", "0") not in ("1", "true")


# Query rows per Gluon prefill program: msa_fwd's prefill BLOCK_M (32) // 16.
PREFILL_SORT_TOPK_Q = 2
_SORT_TOPK_BUFS: dict = {}
_SORT_TOPK_RETIRED: list = []


def _fused_sort_topk() -> bool:
    return os.getenv("VLLM_ROCM_MINIMAX_M3_K4V8_FUSED_SORT_TOPK", "1") not in ("0", "false")


def minimax_m3_k4v8_prefill_sort_topk(layer: AttentionLayer) -> dict | None:
    """This forward's prefill sort_topk request for the indexer, or None.

    The prefill top-k kwargs: sort_topk_q, the attend's prefill block table
    and query_req / query_pos, and fresh sort_topk_out buffers, which the k4v8
    attend then takes as msa_fwd(sort_topk=...). None: no prefill rows, not the
    Gluon attend, or disabled."""
    if _backend() != "gluon" or not _fused_sort_topk():
        return None
    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return None  # profiling run
    md = attn_metadata[layer.layer_name]  # type: ignore[attr-defined]
    assert isinstance(md, MiniMaxM3SparseMetadata)
    if md.num_prefills == 0:
        return None
    p = md.prefill
    assert p is not None and p.query_req_id is not None and p.query_abs_pos is not None
    topk = layer.topk_indices_buffer  # type: ignore[attr-defined]
    assert topk is not None
    rows = md.num_actual_tokens - md.num_decode_tokens
    q = PREFILL_SORT_TOPK_Q
    heads, n_topk = topk.shape[0], topk.shape[2]
    groups = (rows + q - 1) // q
    c = q * n_topk
    # One grow-only buffer per (device, stream), shared by the layers: the
    # indexer writes it and the attend reads it within a layer, in stream order.
    dev = topk.device
    n = heads * groups * c
    key = (dev, torch.cuda.current_stream(dev).cuda_stream)
    buf = _SORT_TOPK_BUFS.get(key)
    need = n + heads * groups + n * q + groups * q
    if buf is None or buf.numel() < need:
        if buf is not None:
            _SORT_TOPK_RETIRED.append(buf)  # never free: a captured graph may hold it
        buf = _SORT_TOPK_BUFS[key] = torch.empty(max(need, 2 * (0 if buf is None else buf.numel())), dtype=torch.int32, device=dev)
    o1 = n + heads * groups
    o2 = o1 + n * q
    out = (
        buf[:n].view(heads, groups, c),
        buf[n:o1].view(heads, groups),
        buf[o1:o2].view(heads, groups, c, q),
        buf[o2 : o2 + groups * q],
    )
    return dict(
        sort_topk_q=q,
        block_table=p.block_table,
        query_req=p.query_req_id,
        query_pos=p.query_abs_pos,
        sort_topk_out=out,
    )


def k4v8_records(kv_cache: torch.Tensor) -> torch.Tensor:
    """The bound [pages, Hkv, 128, 196] uint8 cache as AITER's
    [pages, Hkv, 128 * 196] records (each (page, head) is contiguous)."""
    return kv_cache.flatten(2)


class MiniMaxM3SparseRocmK4v8Backend(MiniMaxM3SparseRocmNvfp4Backend):
    """MiniMax M3 sparse attend over k4v8 K/V pages (ROCm, gfx950).

    Reuses the NVFP4 backend's metadata builder (prefill token positions)."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["nvfp4"]

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        """One slot per KV head holding K and V of a token in 196 bytes."""
        if spec.state_content_bytes is not None:
            return spec
        return replace(
            spec,
            num_head_slots=spec.num_kv_heads,
            state_content_bytes=K4V8_TOKEN_BYTES,
        )


class MiniMaxM3SparseRocmK4v8Impl(MiniMaxM3SparseImpl):
    """AITER MSA (Gluon or Triton) decode and prefill over the k4v8 cache."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        reason = rocm_nvfp4_unsupported_reason(
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            self.topk_blocks,
            self.block_size,
        )
        if reason is not None:
            raise ValueError(f"MiniMax M3 k4v8 KV cache on ROCm {reason}")
        self.q_mode = _q_mode()
        self.p_fp8 = _p_fp8()
        self.backend = _backend()
        if self.backend == "gluon":
            # The model's decode-only MXFP8 hook (hasattr-gated): only the Gluon
            # attend has the MXFP8 epilogue.
            self.forward_decode_mxfp8 = self._forward_decode_mxfp8

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor | None,
        quant_out: tuple[torch.Tensor, torch.Tensor] | None = None,
        prefill_sort_topk: tuple | None = None,
        decode_only: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Attend into output (bf16), or, given quant_out (e4m3 codes
        [tokens, H x 128], E8M0 scales [tokens, H x 4]) on a decode-only Gluon
        step covering every query row, into those as o_proj's MXFP8 input
        (msa_fwd(out_mxfp8=...), bit-exact with mxfp8_e4m3_quantize of the bf16
        output); returns the tensors written. prefill_sort_topk: the prefill
        rows' sort_topk the indexer emitted (minimax_m3_k4v8_prefill_sort_topk's
        sort_topk_out), or None for msa_fwd to build it. decode_only: the caller
        (forward_decode_mxfp8) guarantees a decode-only step; quant_out then
        takes the decode rows even when query has more (CUDA graph padding)."""
        from aiter.ops.triton.attention.msa import msa_fwd

        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return output  # profiling run; caches unbound
        md = attn_metadata[layer.layer_name]  # type: ignore[attr-defined]
        assert isinstance(md, MiniMaxM3SparseMetadata)

        nd = md.num_decode_tokens
        num_tokens = md.num_actual_tokens
        topk = layer.topk_indices_buffer  # type: ignore[attr-defined]
        assert topk is not None
        out = None if output is None else output[:num_tokens].view(-1, self.num_heads, self.head_size)
        # MXFP8 epilogue on decode-only steps whose rows are all the layer's rows
        # (as the NVFP4 attend: mixed steps keep bf16).
        fuse_q = (
            quant_out is not None
            and self.backend == "gluon"
            and md.num_prefills == 0
            and md.num_decodes > 0
            and quant_out[0].shape[0] >= nd
            and (decode_only or (nd == query.shape[0] and quant_out[0].shape[0] == nd))
        )
        cache = k4v8_records(kv_cache)
        v_scale = layer._v_scale  # type: ignore[attr-defined]
        kw = dict(q_mode=self.q_mode, p_fp8=self.p_fp8, backend=self.backend)
        # MXFP8 q straight from the fused insert ([T, H * 128] e4m3 + E8M0
        # scales), or bf16 q [T, H, 128] the attend quantizes itself.
        q_scale = None
        if query.dtype == torch.float8_e4m3fn:
            q = query[:num_tokens]
            q_scale = layer._k4v8_q_scale[:num_tokens]  # type: ignore[attr-defined]
        else:
            q = query[:num_tokens].view(-1, self.num_heads, self.head_size)

        def scale_rows(lo, hi):
            return None if q_scale is None else dict(q_scale=q_scale[lo:hi])

        if md.num_decodes > 0:
            d = md.decode
            assert d is not None
            msa_fwd(
                q[:nd],
                cache,
                topk[:, :nd, :],
                d.block_table,
                self.scale,
                v_scale,
                seq_lens=d.seq_lens,
                decode_query_len=d.decode_query_len,
                out=None if out is None else out[:nd],
                out_mxfp8=(quant_out[0][:nd], quant_out[1][:nd]) if fuse_q else None,
                # Mixed steps run eagerly and are host-bound: one launch (the last
                # split merges) instead of main + reduce. Decode-only steps replay
                # CUDA graphs, where the two-kernel path is faster on the GPU.
                use_atomic_add=md.num_prefills > 0,
                **kw,
                **(scale_rows(0, nd) or {}),
            )
            if fuse_q:
                return quant_out
        if md.num_prefills > 0:
            p = md.prefill
            assert p is not None
            assert p.query_req_id is not None and p.query_abs_pos is not None
            msa_fwd(
                q[nd:],
                cache,
                topk[:, nd:num_tokens, :],
                p.block_table,
                self.scale,
                v_scale,
                query_req=p.query_req_id,
                query_pos=p.query_abs_pos,
                out=out[nd:],
                sort_topk=prefill_sort_topk,
                **kw,
                **(scale_rows(nd, num_tokens) or {}),
            )
        return output

    def _forward_decode_mxfp8(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        out_q: torch.Tensor,
        out_scale: torch.Tensor,
    ) -> None:
        """forward_decode_mxfp8 (Gluon only): a decode-only batch with the
        output written as o_proj's MXFP8 input (out_q e4m3 [tokens, H * 128],
        out_scale E8M0 [tokens, H * 4]), equal to mxfp8_e4m3_quantize of the
        bf16 output."""
        res = self.forward(
            layer, query, kv_cache, None, quant_out=(out_q, out_scale), decode_only=True
        )
        assert isinstance(res, tuple), "decode-only MXFP8 step did not take the epilogue"


def minimax_m3_k4v8_q_is_mxfp8() -> bool:
    """Whether the insert hands the attend MXFP8 q (the Gluon single-term path)."""
    return _backend() == "gluon" and _q_mode() == "mxfp8"


def minimax_m3_norm_rope_k4v8_insert(layer, qkv, positions, slot_mapping, index_slot_mapping):
    """Norm + RoPE of q / k / index-q / index-k, the k4v8 K/V insert and the
    NVFP4 index-K insert in one launch (AITER's
    fused_minimax_m3_norm_rope_quant_insert); returns (q, index_q).

    q, index_q and the index cache are byte-identical to the NVFP4 path's
    FlyDSL insert, so the indexer sees exactly what it does there. With the
    Gluon single-term attend q comes back as MXFP8 (e4m3 [T, q_size]), its
    E8M0 scales left on the layer for the attend.
    """
    from aiter.ops.triton.quant.fused_mxfp8_quant import (
        fused_minimax_m3_norm_rope_quant_insert,
    )

    indexer = layer.indexer
    assert indexer is not None
    num_tokens = qkv.shape[0]
    index_q = qkv.new_empty(
        (num_tokens, layer.index_q_size), dtype=indexer.index_cache.query_dtype
    )
    out_quant = minimax_m3_k4v8_q_is_mxfp8()
    q, q_scale, index_q = fused_minimax_m3_norm_rope_quant_insert(
        qkv,
        layer.q_norm.weight,
        layer.k_norm.weight,
        layer.index_q_norm.weight,
        layer.index_k_norm.weight,
        layer.rotary_emb.cos_sin_cache,
        positions,
        slot_mapping,
        index_slot_mapping,
        layer.kv_cache,
        indexer.index_cache.kv_cache,
        layer._v_scale,
        num_heads=layer.num_heads,
        num_kv_heads=layer.num_kv_heads,
        num_index_heads=layer.num_idx_heads,
        eps=layer.q_norm.variance_epsilon,
        out_quant=out_quant,
        index_q_out=index_q,
    )
    layer._k4v8_q_scale = q_scale
    return q, index_q
