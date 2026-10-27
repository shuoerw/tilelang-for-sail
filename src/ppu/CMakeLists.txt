# PPU: PPU backend
# PPU registers its own "ppu" target kind (device type kDLPPU) so that routing
# and codegen dispatch happen by kind name, mirroring how ROCm uses "hip".


if(NOT USE_PPU)
  return()
endif()

message(STATUS "PPU Backend is enabled")

# PPU: resolve PPU SDK root from PPU_SDK env var, default to /usr/local/PPU_SDK
set(PPU_SDK_ROOT "$ENV{PPU_SDK}" CACHE PATH "PPU SDK root directory")
if(NOT PPU_SDK_ROOT)
  set(PPU_SDK_ROOT "/usr/local/PPU_SDK")
endif()
message(STATUS "PPU SDK root: ${PPU_SDK_ROOT}")

# PPU: add PPU SDK include and library paths for hggc driver/runtime
include_directories(${PPU_SDK_ROOT}/include)
include_directories(${PPU_SDK_ROOT}/targets/x86_64-linux/include)
link_directories(${PPU_SDK_ROOT}/targets/x86_64-linux/lib)

# Export link libraries via the generic TILELANG_ACTIVE_BACKEND_STUB_LINK mechanism
list(APPEND TILELANG_ACTIVE_BACKEND_STUB_LINK hggc hggcrt1)

file(GLOB TILE_LANG_PPU_TRANSFORM_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ppu/transform/*.cc"
)

# PPU: register the standalone "ppu" target kind, codegen, op lowering, and transform passes.
list(APPEND TILE_LANG_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ppu/target/target_ppu.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ppu/target_utils.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ppu/codegen/codegen_ppu.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ppu/codegen/intrin_rule_ppu.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ppu/codegen/tix.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ppu/codegen/rt_mod_ppu.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ppu/runtime/ppu_device_api.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ppu/runtime/runtime.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ppu/op/atomic_add.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ppu/op/atomic_reduce.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ppu/op/builtin.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ppu/op/copy.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ppu/op/copy_analysis.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ppu/op/fill.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ppu/op/finalize_reducer.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ppu/op/gemm.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ppu/op/gemm_sp.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ppu/op/reduce.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ppu/op/scan.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ppu/op/transpose.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ppu/layout/ppu_gemm_layouts.cc"
  ${TILE_LANG_PPU_TRANSFORM_SRCS}
)
