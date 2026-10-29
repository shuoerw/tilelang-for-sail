"""PPU dialect loop constructs with PPU-native lowering hints."""

from __future__ import annotations

from typing import Any

from tvm import tirx
from tvm.tirx.script.builder import frame

from tilelang.language.loop import (
    Parallel as _common_Parallel,
    unroll as _common_unroll,
)

__all__ = ["Parallel", "Unroll", "unroll"]


def Parallel(
    *extents: int | tirx.PrimExpr,
    coalesced_width: int | None = None,
    loop_layout: Any | None = None,
    prefer_async: bool | None = None,
    annotations: dict[str, Any] | None = None,
) -> frame.ForFrame:
    """Construct a nested parallel loop with the PPU async-copy hint.

    ``prefer_async=True`` requests async-copy injection for this loop subtree.
    The hint is lowered as the ``"parallel_prefer_async"`` annotation.
    Values supplied directly in ``annotations`` take precedence.
    """
    ann: dict[str, Any] = dict(annotations) if annotations is not None else {}
    if prefer_async is not None:
        ann.setdefault("parallel_prefer_async", prefer_async)
    return _common_Parallel(
        *extents,
        coalesced_width=coalesced_width,
        loop_layout=loop_layout,
        annotations=ann or None,
    )


def unroll(
    start: tirx.PrimExpr,
    stop: tirx.PrimExpr | None = None,
    step: tirx.PrimExpr | None = None,
    *,
    explicit: bool = False,
    unroll_factor: int | None = None,
    annotations: dict[str, Any] | None = None,
) -> frame.ForFrame:
    """Construct an unrolled loop with the PPU unroll-factor pragma."""
    ann: dict[str, Any] = dict(annotations) if annotations is not None else {}
    if unroll_factor is not None:
        ann.setdefault("pragma_unroll_factor", unroll_factor)
    return _common_unroll(start, stop, step, explicit=explicit, annotations=ann or None)


def Unroll(
    start: tirx.PrimExpr,
    stop: tirx.PrimExpr | None = None,
    step: tirx.PrimExpr | None = None,
    *,
    explicit: bool = False,
    unroll_factor: int | None = None,
    annotations: dict[str, Any] | None = None,
) -> frame.ForFrame:
    """Alias of the PPU dialect's :func:`unroll`."""
    return unroll(
        start,
        stop,
        step,
        explicit=explicit,
        unroll_factor=unroll_factor,
        annotations=annotations,
    )
