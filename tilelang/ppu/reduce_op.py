"""PPU reduction operators with PPU-native lowering hints."""

from __future__ import annotations

from tvm import tirx

from tilelang.language.reduce_op import (
    reduce_absmax as _common_reduce_absmax,
    reduce_max as _common_reduce_max,
    reduce_min as _common_reduce_min,
)

__all__ = ["reduce_absmax", "reduce_max", "reduce_min"]


def _with_nan_propagate(annotations: dict | None, nan_propagate: bool) -> dict | None:
    if not nan_propagate:
        return annotations
    ann = dict(annotations) if annotations is not None else {}
    ann.setdefault("nan_propagate", True)
    return ann


def reduce_max(
    buffer: tirx.Buffer,
    out: tirx.Buffer,
    dim: int = -1,
    clear: bool = True,
    batch: int = 1,
    nan_propagate: bool = False,
    annotations: dict | None = None,
) -> None:
    """Perform reduce max with the PPU NaN-propagation hint."""
    _common_reduce_max(
        buffer,
        out,
        dim,
        clear,
        batch=batch,
        annotations=_with_nan_propagate(annotations, nan_propagate),
    )


def reduce_min(
    buffer: tirx.Buffer,
    out: tirx.Buffer,
    dim: int = -1,
    clear: bool = True,
    batch: int = 1,
    nan_propagate: bool = False,
    annotations: dict | None = None,
) -> None:
    """Perform reduce min with the PPU NaN-propagation hint."""
    _common_reduce_min(
        buffer,
        out,
        dim,
        clear,
        batch=batch,
        annotations=_with_nan_propagate(annotations, nan_propagate),
    )


def reduce_absmax(
    buffer: tirx.Buffer,
    out: tirx.Buffer,
    dim: int = -1,
    clear: bool = True,
    batch: int = 1,
    nan_propagate: bool = False,
    annotations: dict | None = None,
) -> None:
    """Perform reduce absolute max with the PPU NaN-propagation hint."""
    _common_reduce_absmax(
        buffer,
        out,
        dim,
        clear,
        batch=batch,
        annotations=_with_nan_propagate(annotations, nan_propagate),
    )
