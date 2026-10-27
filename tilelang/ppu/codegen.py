from __future__ import annotations

from tilelang.backend.device_codegen import global_func_device_codegen

build_ppu = global_func_device_codegen("target.build.tilelang_ppu")
build_ppu_without_compile = global_func_device_codegen("target.build.tilelang_ppu_without_compile")
