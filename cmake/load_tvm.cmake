# todo: support prebuilt tvm

set(TVM_BUILD_FROM_SOURCE TRUE)
set(TVM_SOURCE ${CMAKE_SOURCE_DIR}/3rdparty/tvm)

if(DEFINED ENV{TVM_ROOT})
  if(EXISTS $ENV{TVM_ROOT}/cmake/config.cmake)
    set(TVM_SOURCE $ENV{TVM_ROOT})
    message(STATUS "Using TVM_ROOT from environment variable: ${TVM_SOURCE}")
  endif()
endif()

message(STATUS "Using TVM source: ${TVM_SOURCE}")

set(TVM_INCLUDES
  ${TVM_SOURCE}/include
  ${TVM_SOURCE}/src
  ${TVM_SOURCE}/3rdparty/dlpack/include
)

if(EXISTS ${TVM_SOURCE}/ffi/include)
  list(APPEND TVM_INCLUDES ${TVM_SOURCE}/ffi/include)
elseif(EXISTS ${TVM_SOURCE}/3rdparty/tvm-ffi/include)
  list(APPEND TVM_INCLUDES ${TVM_SOURCE}/3rdparty/tvm-ffi/include)
endif()

if(EXISTS ${TVM_SOURCE}/3rdparty/tvm-ffi/3rdparty/dlpack/include)
  list(APPEND TVM_INCLUDES ${TVM_SOURCE}/3rdparty/tvm-ffi/3rdparty/dlpack/include)
endif()

# --- Patch 0: Linker.cmake — replace -fuse-ld=lld with -fuse-ld=gold ---
set(linker_cmake "${TVM_SOURCE}/cmake/utils/Linker.cmake")
if(EXISTS "${linker_cmake}")
  file(READ "${linker_cmake}" FILE_CONTENTS)
  if(FILE_CONTENTS MATCHES "-fuse-ld=lld")
    string(REPLACE "-fuse-ld=lld" "-fuse-ld=gold" NEW_CONTENTS "${FILE_CONTENTS}")
    file(WRITE "${linker_cmake}" "${NEW_CONTENTS}")
    message(STATUS "Patched Linker.cmake: replaced -fuse-ld=lld with -fuse-ld=gold")
  endif()
endif()

# --- Patch 1: dlpack.h — add kDLPPU device type ---
set(dlpack_header "${TVM_SOURCE}/3rdparty/tvm-ffi/3rdparty/dlpack/include/dlpack/dlpack.h")
if(EXISTS "${dlpack_header}")
  file(READ "${dlpack_header}" FILE_CONTENTS)
  if(NOT FILE_CONTENTS MATCHES ".*kDLPPU.*")
    string(REPLACE
      "} DLDeviceType;"
      "  /*! \\brief T-HEAD PPU device */\n  kDLPPU = 19,\n} DLDeviceType;"
      NEW_CONTENTS "${FILE_CONTENTS}")
    file(WRITE "${dlpack_header}" "${NEW_CONTENTS}")
    message(STATUS "Patched dlpack.h: added kDLPPU = 19")
  endif()
endif()

# --- Patch 2: device_api.h — add kDLPPU string mapping ---
set(tvm_device_api_header "${TVM_SOURCE}/include/tvm/runtime/device_api.h")
if(EXISTS "${tvm_device_api_header}")
  file(READ "${tvm_device_api_header}" FILE_CONTENTS)
  if(NOT FILE_CONTENTS MATCHES ".*kDLPPU.*")
    string(REPLACE
      "      return \"hexagon\";\n    default:"
      "      return \"hexagon\";\n    case kDLPPU:\n      return \"ppu\";\n    default:"
      NEW_CONTENTS "${FILE_CONTENTS}")
    file(WRITE "${tvm_device_api_header}" "${NEW_CONTENTS}")
    message(STATUS "Patched device_api.h: added kDLPPU string mapping")
  endif()
endif()



# --- Patch 3: tensor.h --- add kDLPPU to IsDirectAddressDevice ---
set(tvm_ffi_tensor_h "${TVM_SOURCE}/3rdparty/tvm-ffi/include/tvm/ffi/container/tensor.h")
if(EXISTS "${tvm_ffi_tensor_h}")
  file(READ "${tvm_ffi_tensor_h}" FILE_CONTENTS)
  if(NOT FILE_CONTENTS MATCHES ".*kDLPPU.*")
    string(REPLACE
      "device.device_type == kDLROCMHost;"
      "device.device_type == kDLROCMHost ||
         device.device_type == kDLPPU;"
      NEW_CONTENTS "${FILE_CONTENTS}")
    file(WRITE "${tvm_ffi_tensor_h}" "${NEW_CONTENTS}")
    message(STATUS "Patched tensor.h: added kDLPPU to IsDirectAddressDevice")
  endif()
endif()


# --- Patch 4: env_context.cc --- kDLCUDA to kDLPPU stream redirect ---
# PPU-only build: redirect kDLCUDA stream ops to kDLPPU
set(tvm_ffi_env_context "${TVM_SOURCE}/3rdparty/tvm-ffi/src/ffi/extra/env_context.cc")
if(USE_PPU AND NOT USE_CUDA AND EXISTS "${tvm_ffi_env_context}")
  file(READ "${tvm_ffi_env_context}" FILE_CONTENTS)
  if(NOT FILE_CONTENTS MATCHES ".*kDLPPU.*")
    string(REPLACE
      "tvm::ffi::EnvContext::ThreadLocal()->SetStream(device_type, device_id, stream,"
      "if (device_type == DLDeviceType::kDLCUDA) {\n    device_type = (int32_t)DLDeviceType::kDLPPU;\n  }\n  tvm::ffi::EnvContext::ThreadLocal()->SetStream(device_type, device_id, stream,"
      NEW_CONTENTS "${FILE_CONTENTS}")
    string(REPLACE
      "return tvm::ffi::EnvContext::ThreadLocal()->GetStream(device_type, device_id);"
      "if (device_type == DLDeviceType::kDLCUDA) {\n    device_type = (int32_t)DLDeviceType::kDLPPU;\n  }\n  return tvm::ffi::EnvContext::ThreadLocal()->GetStream(device_type, device_id);"
      NEW_CONTENTS "${NEW_CONTENTS}")
    file(WRITE "${tvm_ffi_env_context}" "${NEW_CONTENTS}")
    message(STATUS "Patched env_context.cc: kDLCUDA -> kDLPPU stream redirect")
  endif()
endif()

# Scope positive loop-extent assumptions to the loop body, preserving zero-trip loops.
# Backport: https://github.com/tile-ai/tvm/pull/86
# Apache TVM source: 434dc7636e06c6c4a25eb1845869b596199c51bb
function(tilelang_patch_tvm_loop_analysis relative_path before after)
  set(source_file "${TVM_SOURCE}/${relative_path}")
  file(READ "${source_file}" contents)
  string(FIND "${contents}" "${after}" patched_position)
  if(NOT patched_position EQUAL -1)
    return()
  endif()
  string(FIND "${contents}" "${before}" original_position)
  if(original_position EQUAL -1)
    message(FATAL_ERROR
      "Cannot apply TVM loop-analysis fix to ${source_file}: unexpected source. "
      "Check compatibility with https://github.com/tile-ai/tvm/pull/86")
  endif()
  string(REPLACE "${before}" "${after}" contents "${contents}")
  file(WRITE "${source_file}" "${contents}")
  message(STATUS "Patched ${relative_path}: scope positive loop extents to the body")
endfunction()

tilelang_patch_tvm_loop_analysis("src/arith/ir_mutator_with_analyzer.cc"
[=[    constraint_scope_.Current().Emplace(analyzer_, op->extent > 0);
    return StmtExprMutator::VisitStmt_(op);]=]
[=[

    PrimExpr min = this->VisitExpr(op->min);
    PrimExpr extent = this->VisitExpr(op->extent);
    ffi::Optional<PrimExpr> step{std::nullopt};
    if (op->step.has_value()) {
      step = this->VisitExpr(*op->step);
    }
    Stmt body = constraint_scope_.WithNewScope([&]() -> Stmt {
      constraint_scope_.Current().Emplace(analyzer_, extent > 0);
      return this->VisitStmt(op->body);
    });
    if (min.same_as(op->min) && extent.same_as(op->extent) && body.same_as(op->body) &&
        step.same_as(op->step)) {
      return ffi::GetRef<Stmt>(op);
    } else {
      auto n = this->CopyOnWrite(op);
      n->min = std::move(min);
      n->extent = std::move(extent);
      n->step = std::move(step);
      n->body = std::move(body);
      return Stmt(n);
    }]=])

tilelang_patch_tvm_loop_analysis("src/arith/ir_visitor_with_analyzer.cc"
[=[    constraint_scope_.Current().Emplace(&analyzer_, op->extent > 0);
    StmtExprVisitor::VisitStmt_(op);]=]
[=[    this->VisitExpr(op->min);
    this->VisitExpr(op->extent);
    if (op->step.has_value()) {
      this->VisitExpr(*op->step);
    }
    constraint_scope_.WithNewScope([&]() {
      constraint_scope_.Current().Emplace(&analyzer_, op->extent > 0);
      this->VisitStmt(op->body);
    });]=])
