# PPU backend: source files, toolchain, and build configuration.
#
# Target registration, target helpers, intrinsic rules, and builtin ops are pure
# C++ dependencies used by common code. Compile them even when the PPU SDK-backed
# runtime and device codegen are disabled.
file(GLOB TILE_LANG_PPU_ALWAYS_SRCS
  src/ppu/codegen/intrin_rule_ppu.cc
  src/ppu/op/builtin.cc
  src/ppu/target/target_ppu.cc
  src/ppu/target_utils.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_PPU_ALWAYS_SRCS})

if(NOT USE_PPU)
  return()
endif()

message(STATUS "PPU Backend is enabled")

# USE_PPU accepts ON or an explicit SDK root. For ON, prefer PPU_SDK from the
# environment and retain the historical /usr/local/PPU_SDK fallback.
if(NOT USE_PPU STREQUAL "ON" AND NOT USE_PPU STREQUAL "TRUE" AND
   NOT USE_PPU STREQUAL "1")
  set(_TILELANG_PPU_SDK_DEFAULT "${USE_PPU}")
elseif(DEFINED ENV{PPU_SDK} AND NOT "$ENV{PPU_SDK}" STREQUAL "")
  set(_TILELANG_PPU_SDK_DEFAULT "$ENV{PPU_SDK}")
else()
  set(_TILELANG_PPU_SDK_DEFAULT "/usr/local/PPU_SDK")
endif()
set(PPU_SDK_ROOT "${_TILELANG_PPU_SDK_DEFAULT}" CACHE PATH
    "PPU SDK root directory")
unset(_TILELANG_PPU_SDK_DEFAULT)
message(STATUS "PPU SDK root: ${PPU_SDK_ROOT}")

list(APPEND TILE_LANG_INCLUDES
  "${PPU_SDK_ROOT}/include"
  "${PPU_SDK_ROOT}/targets/x86_64-linux/include"
)
link_directories("${PPU_SDK_ROOT}/targets/x86_64-linux/lib")

file(GLOB TILE_LANG_PPU_ACTIVE_SRCS
  src/ppu/codegen/*.cc
  src/ppu/layout/*.cc
  src/ppu/op/*.cc
  src/ppu/runtime/*.cc
  src/ppu/transform/*.cc
)
list(REMOVE_ITEM TILE_LANG_PPU_ACTIVE_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ppu/codegen/intrin_rule_ppu.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ppu/op/builtin.cc"
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_PPU_ACTIVE_SRCS})

# The PPU runtime and generated modules link against the SDK driver/runtime.
list(APPEND TILELANG_ACTIVE_BACKEND_STUB_LINK hggc hggcrt1)
