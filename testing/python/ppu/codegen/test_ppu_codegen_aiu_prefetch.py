"""Validate per-load AIU prefetch on PPU 1.5 without requiring a device."""

import os
from pathlib import Path
import re
import subprocess

import pytest

import tilelang.ppu.language as T
from tilelang import tvm
from tilelang.engine.lower import lower
from tilelang.ppu.backend import tilelang_callback_ppu_compile
from tilelang.layout import make_full_bank_swizzled_layout


TARGET = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})


def _call(**kwargs):
    return T.ppu_aiu_load(
        tvm.tirx.Var("shared", "handle"),
        tvm.tirx.Var("global", "handle"),
        128,
        16,
        128,
        16,
        128,
        0,
        0,
        0,
        **kwargs,
    )


@pytest.mark.parametrize("prefetch_bytes", [None, 0, 64, 128, 256, tvm.tirx.IntImm("int32", 64)])
def test_aiu_prefetch_ir(prefetch_bytes):
    kwargs = {} if prefetch_bytes is None else {"prefetch_bytes": prefetch_bytes}
    call = _call(**kwargs)
    assert len(call.args) == 12
    assert int(call.args[11]) == (128 if prefetch_bytes is None else int(prefetch_bytes))


@pytest.mark.parametrize("prefetch_bytes", [-1, 32, 512])
def test_aiu_prefetch_rejects_invalid_size(prefetch_bytes):
    with pytest.raises(ValueError, match="prefetch_bytes must be 0, 64, 128, or 256"):
        _call(prefetch_bytes=prefetch_bytes)


@pytest.mark.parametrize("prefetch_bytes", [True, False, 128.0, "128", tvm.tirx.Var("size", "int32")])
def test_aiu_prefetch_requires_compile_time_integer(prefetch_bytes):
    with pytest.raises(TypeError, match="prefetch_bytes must be a compile-time integer"):
        _call(prefetch_bytes=prefetch_bytes)


def _kernel(prefetch_bytes=None, mixed=False, use_copy=False, is_swizzle=1):
    kwargs = {} if prefetch_bytes is None else {"prefetch_bytes": prefetch_bytes}

    @T.prim_func
    def main(A: T.Tensor((16, 64), "float16"), B: T.Tensor((16, 64), "float16")):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((16, 64), "float16")
            if is_swizzle:
                T.annotate_layout({shared: make_full_bank_swizzled_layout(shared)})
            if use_copy:
                T.copy(A, shared, prefer_instruction="aiu")
            else:
                T.ppu_aiu_load(
                    T.access_ptr(shared[0, 0], "w", extent=1024),
                    T.access_ptr(A[0, 0], "r", extent=1024),
                    128,
                    16,
                    128,
                    16,
                    128,
                    0,
                    0,
                    0,
                    is_swizzle,
                    **kwargs,
                )
                T.ptx_commit_group()
                T.ptx_wait_group(0)
            if mixed:
                T.ppu_aiu_load(
                    T.access_ptr(shared[0, 0], "w", extent=1024),
                    T.access_ptr(A[0, 0], "r", extent=1024),
                    128,
                    16,
                    128,
                    16,
                    128,
                    0,
                    0,
                    0,
                    is_swizzle,
                    prefetch_bytes=256,
                )
                T.ptx_commit_group()
                T.ptx_wait_group(0)
            T.copy(shared, B)

    return main


def _source(program):
    with TARGET, tvm.transform.PassContext(opt_level=3, config={"tl.disable_aiu_lower": False}):
        return lower(program, target=TARGET).kernel_source


def _prefetch_arg(prefetch_bytes):
    mode = "kNone" if prefetch_bytes == 0 else f"kLLC{prefetch_bytes}B"
    return f"tl::AiuPrefetch::{mode}"


@pytest.mark.parametrize("prefetch_bytes", [None, 0, 64, 128, 256])
@pytest.mark.parametrize("is_swizzle", [0, 1])
def test_aiu_prefetch_codegen(prefetch_bytes, is_swizzle):
    source = _source(_kernel(prefetch_bytes, is_swizzle=is_swizzle))
    effective_bytes = 128 if prefetch_bytes is None else prefetch_bytes
    helper = "tl::aiu_load(" if is_swizzle else "tl::aiu_load<false>("
    assert helper in source
    assert f", {_prefetch_arg(effective_bytes)});" in source
    assert source.count("tl::aiu_load") == 1


def test_aiu_prefetch_can_differ_within_kernel():
    source = _source(_kernel(0, mixed=True))
    assert f", {_prefetch_arg(0)});" in source
    assert f", {_prefetch_arg(256)});" in source
    assert source.count("tl::aiu_load") == 2


def _rewrite_prefetch(program, value):
    def rewrite(node):
        if isinstance(node, tvm.tirx.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "tl.ppu_aiu_load":
            args = [*node.args[:11], value]
            return tvm.tirx.Call(node.dtype, node.op, args, annotations=node.annotations)
        return None

    return program.with_body(tvm.tirx.stmt_functor.ir_transform(program.body, None, rewrite, ["tirx.Call"]))


def test_copy_uses_default_aiu_prefetch():
    source = _source(_kernel(use_copy=True))
    assert f", {_prefetch_arg(128)});" in source
    assert source.count("tl::aiu_load") == 1


@pytest.mark.parametrize("value", [tvm.tirx.IntImm("int32", 32), tvm.tirx.Var("size", "int32")])
def test_aiu_prefetch_codegen_validates_raw_ir(value):
    program = _rewrite_prefetch(_kernel(), value)
    if isinstance(value, tvm.tirx.Var):
        program = tvm.tirx.PrimFunc(
            list(program.params) + [value],
            program.body,
            buffer_map=program.buffer_map,
            attrs=program.attrs,
        )
    with pytest.raises(tvm.error.InternalError, match="ppu_aiu_load prefetch_bytes must be"):
        _source(program)


@pytest.fixture(scope="module")
def sdk_tools():
    sdk = os.environ.get("PPU_SDK")
    if not sdk or not Path(sdk, "bin", "hgcc").is_file():
        pytest.skip("PPU SDK compiler is required; no PPU device is needed")
    tools = (Path(sdk, "bin", "llvm-dis"), Path(sdk, "bin", "llvm-objdump"))
    if not all(tool.is_file() for tool in tools):
        pytest.skip("PPU SDK llvm-dis and llvm-objdump are required to inspect prefetch instructions")
    return tools


def _assemble(source, tmp_path, tools):
    binary = tilelang_callback_ppu_compile(
        source,
        TARGET,
        {"tl.device_compile_flags": ["-keep", "--keep-dir", str(tmp_path)]},
    )
    assert binary
    bitcode = list(tmp_path.glob("*ppu-ppu0015.bc"))
    objects = list(tmp_path.glob("*ppu0015.o"))
    assert len(bitcode) == 1
    assert len(objects) == 1
    result = subprocess.run([str(tools[0]), str(bitcode[0]), "-o", "-"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    # Frontend IR retains switch arms; disassembly below checks the selected instruction.
    instructions = re.findall(r"ppu\.cp\.async\.aiu\.bulk\.[^ ]+", result.stdout)
    result = subprocess.run([str(tools[1]), "-d", str(objects[0])], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return instructions, [line for line in result.stdout.splitlines() if "vmem.aiu.ld.tsm." in line]


def _instruction(prefetch_bytes, is_swizzle=1):
    hint = f".LLC::{prefetch_bytes}B" if prefetch_bytes else ""
    layout = "swzl" if is_swizzle else "linear"
    return f"ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile{hint}.padz.{layout}.b8"


@pytest.mark.parametrize("prefetch_bytes", [None, 0, 64, 128, 256])
@pytest.mark.parametrize("is_swizzle", [0, 1])
def test_aiu_prefetch_assembled_instruction(sdk_tools, tmp_path, prefetch_bytes, is_swizzle):
    instructions, lines = _assemble(_source(_kernel(prefetch_bytes, is_swizzle=is_swizzle)), tmp_path, sdk_tools)
    assert len(lines) == 1
    effective_bytes = 128 if prefetch_bytes is None else prefetch_bytes
    assert _instruction(effective_bytes, is_swizzle) in instructions
    if effective_bytes in (128, 256):
        assert f".pref{effective_bytes}" in lines[0]
    elif effective_bytes == 64:
        # The 64B encoding can omit a printed prefetch suffix.
        assert ".pref" not in lines[0] or ".pref64" in lines[0]
    else:
        assert ".pref" not in lines[0]


def test_aiu_prefetch_mixed_assembled_instructions(sdk_tools, tmp_path):
    instructions, lines = _assemble(_source(_kernel(0, mixed=True)), tmp_path, sdk_tools)
    assert {_instruction(0), _instruction(256)}.issubset(instructions)
    assert len(lines) == 2
    assert ".pref" not in lines[0]
    assert ".pref256" in lines[1]


def test_copy_assembles_default_aiu_prefetch(sdk_tools, tmp_path):
    instructions, lines = _assemble(_source(_kernel(use_copy=True)), tmp_path, sdk_tools)
    assert _instruction(128) in instructions
    assert len(lines) == 1
    assert ".pref128" in lines[0]


@pytest.mark.parametrize("is_swizzle", [0, 1])
def test_aiu_prefetch_helper_default(sdk_tools, tmp_path, is_swizzle):
    source = _source(_kernel(is_swizzle=is_swizzle)).replace(f", {_prefetch_arg(128)}", "")
    instructions, lines = _assemble(source, tmp_path, sdk_tools)
    assert _instruction(128, is_swizzle) in instructions
    assert len(lines) == 1
    assert ".pref128" in lines[0]


@pytest.mark.parametrize("is_swizzle", [0, 1])
def test_aiu_prefetch_helper_accepts_runtime_enum(sdk_tools, tmp_path, is_swizzle):
    source = f"""
#include <tl_templates/ppu/copy.h>

extern "C" __global__ void aiu_prefetch(const unsigned char *input,
                                     tl::AiuPrefetch prefetch) {{
  __shared__ __align__(128) unsigned char shared[1024];
  tl::aiu_load<{"true" if is_swizzle else "false"}>(
      shared, input, 128, 8, 128, 8, 128, 0, 0, 0, prefetch);
  tl::cp_async_commit();
  tl::cp_async_wait<0>();
}}
"""
    instructions, lines = _assemble(source, tmp_path, sdk_tools)
    assert sorted(instructions) == sorted(_instruction(size, is_swizzle) for size in (0, 64, 128, 256))
    assert len(lines) == 4


def test_aiu_prefetch_helper_requires_enum(sdk_tools):
    source = _source(_kernel(128)).replace(_prefetch_arg(128), "128")
    with pytest.raises(RuntimeError, match="no matching function for call to 'aiu_load'"):
        tilelang_callback_ppu_compile(source, TARGET, {})
