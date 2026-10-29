"""PPU dialect of copy operators with backend lowering hints."""

from __future__ import annotations

from typing import Any, Literal

from tvm import tirx

from tilelang._typing import BufferLikeType
from tilelang.language.copy_op import EVICTION_POLICY_IDS, copy as _common_copy

__all__ = ["copy"]


def copy(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    coalesced_width: int | None = None,
    disable_tma: bool = False,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
    prefer_instruction: str | None = None,
    annotations: dict | None = None,
    loop_layout: Any | None = None,
) -> tirx.PrimExpr | tirx.Stmt:
    ann: dict = dict(annotations) if annotations is not None else {}
    if "disable_tma" not in ann and disable_tma:
        ann["disable_tma"] = disable_tma
    if "eviction_policy" not in ann and eviction_policy is not None:
        ann["eviction_policy"] = EVICTION_POLICY_IDS[eviction_policy]
    if "prefer_instruction" not in ann and prefer_instruction is not None:
        ann["prefer_instruction"] = tirx.StringImm(prefer_instruction)
    return _common_copy(
        src,
        dst,
        coalesced_width=coalesced_width,
        annotations=ann or None,
        loop_layout=loop_layout,
    )
