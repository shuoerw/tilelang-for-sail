"""NVRTC Library Generator for TileLang.

Compiles CUDA kernels at runtime using NVRTC and manages resulting binaries.

Why NVRTC instead of nvcc:
- No offline compilation step, enables true JIT workflows
- Works without CUDA toolkit installed (only requires driver)
- Allows kernel specialization based on runtime parameters

Key responsibilities:
- Compile CUDA source to cubin using NVRTC API
- Generate accompanying Python launcher code
- Load compiled cubin and extract kernel handles
- Manage library lifecycle (load/unload)
"""

from __future__ import annotations
import ctypes
import importlib
import logging
import os
import os.path as osp
import tempfile
from types import ModuleType

from tvm.target import Target

from tilelang import tvm as tvm
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.adapter.utils import is_cuda_target, is_ppu_target
from tilelang.jit.adapter.nvrtc import is_nvrtc_available, NVRTC_UNAVAILABLE_MESSAGE
from tilelang.jit.adapter.nvrtc.include_paths import discover_cuda_include_paths

logger = logging.getLogger(__name__)

if is_nvrtc_available:
    import cuda.bindings.driver as cuda
    from tilelang.contrib.nvrtc import compile_cuda, get_nvrtc_version
# Note: no hard failure when cuda-python is missing — the PPU target path of
# this backend (hgcc compile + HGGC driver via ctypes) works without
# cuda-python. CUDA code paths raise naturally if used without it.


_hg_driver_lib = None


def _get_hg_driver():
    """Lazily load libhggc.so and perform one-time hgInit(0).

    HGGC driver must be initialized explicitly before any hgModule*/hgLaunch* call.
    """
    global _hg_driver_lib
    if _hg_driver_lib is not None:
        return _hg_driver_lib

    ppu_sdk = os.environ.get("PPU_SDK")
    if not ppu_sdk:
        raise RuntimeError(
            "PPU_SDK environment variable is not set. Please source the PPU SDK envsetup.sh (e.g. source $PPU_SDK/envsetup.sh ppu)"
        )
    candidates = [
        osp.join(ppu_sdk, "lib", "libhggc.so"),
        osp.join(ppu_sdk, "targets", "x86_64-linux", "lib", "libhggc.so"),
        "libhggc.so",
    ]
    lib = None
    for cand in candidates:
        try:
            lib = ctypes.CDLL(cand)
            break
        except OSError:
            continue
    if lib is None:
        raise RuntimeError(f"Failed to load libhggc.so (tried: {candidates})")

    lib.hgInit.argtypes = [ctypes.c_uint]
    lib.hgInit.restype = ctypes.c_int
    res = lib.hgInit(0)
    if res != 0:
        raise RuntimeError(f"hgInit(0) failed: HGresult={res}")

    # A current context is required before any hgModule*/hgLaunch* call —
    # hgModuleLoadData returns HGGC_ERROR_INVALID_CONTEXT (201) otherwise.
    # Mirrors EnsureCurrentPPUContext in src/ppu/codegen/rt_mod_ppu.cc
    # (hgInit + device context), but stays purely on driver APIs so no
    # second library (libhggcrt/libhggc_wrapper) is needed. The primary
    # context is process-wide per device, so it is shared with torch's
    # PPU runtime allocations.
    lib.hgDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.hgDeviceGet.restype = ctypes.c_int
    lib.hgDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    lib.hgDevicePrimaryCtxRetain.restype = ctypes.c_int
    lib.hgCtxSetCurrent.argtypes = [ctypes.c_void_p]
    lib.hgCtxSetCurrent.restype = ctypes.c_int
    dev = ctypes.c_int(0)
    res = lib.hgDeviceGet(ctypes.byref(dev), 0)
    if res != 0:
        raise RuntimeError(f"hgDeviceGet(0) failed: HGresult={res}")
    ctx = ctypes.c_void_p()
    res = lib.hgDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)
    if res != 0:
        raise RuntimeError(f"hgDevicePrimaryCtxRetain failed: HGresult={res}")
    res = lib.hgCtxSetCurrent(ctx)
    if res != 0:
        raise RuntimeError(f"hgCtxSetCurrent failed: HGresult={res}")
    # Keep the retained primary context referenced for the process lifetime.
    lib._hg_primary_ctx = ctx

    lib.hgModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    lib.hgModuleLoadData.restype = ctypes.c_int
    lib.hgModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    lib.hgModuleGetFunction.restype = ctypes.c_int
    lib.hgModuleUnload.argtypes = [ctypes.c_void_p]
    lib.hgModuleUnload.restype = ctypes.c_int

    _hg_driver_lib = lib
    return lib


class NVRTCLibraryGenerator(LibraryGenerator):
    """Runtime compiler and loader for NVRTC-compiled CUDA kernels.

    Lifecycle:
        1. compile_lib(): CUDA source → cubin + Python launcher
        2. load_lib(): cubin → loaded library + kernel handles
        3. pymodule.call(): Execute kernels via Python launcher
        4. __del__: Cleanup (unload library)

    Why three files (cu, cubin, py):
        - .cu: Source for debugging, kept in temp directory
        - .cubin: Compiled binary, loaded by CUDA driver
        - .py: Launch code, imported as Python module

    Attributes:
        host_func: Generated Python launch code (from wrapper)
        culib: CUDA library handle (CUlibrary)
        pymodule: Imported Python module containing call() function
    """

    host_func: str | None = None
    culib: cuda.CUlibrary | None = None
    pymodule: ModuleType | None = None
    pypath: str | None = None
    hgmod = None  # PPU: HGmodule handle (ctypes.c_void_p) when target is ppu

    def __init__(self, target: Target, verbose: bool = False):
        """Initialize NVRTC library generator.

        Args:
            target: Compilation target (must be CUDA)
            verbose: Enable verbose compilation output
        """
        super().__init__(target, verbose)

    @staticmethod
    def import_from_file(module_name, file_path):
        """Dynamically import Python module from file path.

        Standard importlib pattern for loading modules outside sys.path.
        Used to import generated .py launcher code from temp directory.

        Args:
            module_name: Name to assign to imported module
            file_path: Absolute path to .py file

        Returns:
            Imported module object
        """
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to import module from file: {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def update_host_func(self, host_func: str):
        """Store generated Python launch code for later file write.

        Called by adapter after wrapper generates the launch code.
        This is the bridge between code generation and file output.

        Args:
            host_func: Python source code containing call() function
        """
        self.host_func = host_func

    def load_lib(self, lib_path: str | None = None):
        """Load compiled cubin and Python launcher into memory.

        Why two loads:
            1. Import Python module for launch logic
            2. Load cubin via CUDA Driver API for kernel handles

        Context synchronization: CUDA context must be current before loading.
        If not, use torch.cuda.synchronize() to establish context.

        Args:
            lib_path: Path to .cubin file (optional, uses self.libpath if None)

        Side effects:
            - Sets self.pymodule to imported Python module
            - Sets self.culib to CUDA library handle
        """
        if lib_path is None:
            lib_path = self.libpath
        else:
            self.libpath = lib_path

        if lib_path.endswith(".cubin"):
            self.pypath = lib_path.replace(".cubin", ".py")
        elif lib_path.endswith(".hgbin"):
            self.pypath = lib_path.replace(".hgbin", ".py")
        else:
            self.pypath = lib_path + ".py"
        self.pymodule = self.import_from_file("kernel", self.pypath)

        if is_ppu_target(self.target):
            # PPU: load the hgcc-compiled .hgbin through the HGGC driver.
            self._load_module_ppu(lib_path)
            return

        # Ensure the context is valid
        ctx = cuda.cuCtxGetCurrent()[1]
        if cuda.cuCtxGetApiVersion(ctx)[0] != cuda.CUresult.CUDA_SUCCESS:
            import torch

            torch.cuda.synchronize()

        result, self.culib = cuda.cuLibraryLoadFromFile(bytes(lib_path, "utf-8"), [], [], 0, [], [], 0)
        if result != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Failed to load library: {lib_path}, error: {result}")

    def compile_lib(self, timeout: float | None = None):
        """Compile CUDA source to cubin using NVRTC and write output files.

        Output artifacts (all in temp directory):
            - .cu: Source code (for debugging)
            - .cubin: Compiled binary (for execution)
            - .py: Python launcher (for calling kernels)

        Include paths setup:
            - TileLang templates: kernel primitives and utilities
            - CUTLASS: optimized GEMM/tensor ops
            - CUDA headers: driver/runtime APIs

        Why architecture detection:
            ARM64 servers (SBSA) have different header paths than x86_64.

        Args:
            timeout: Compilation timeout in seconds (currently unsupported by NVRTC compiler)

        Side effects:
            - Writes .cu, .cubin, .py files to temp directory
            - Sets self.srcpath, self.libpath, self.pypath
        """
        target = self.target
        verbose = self.verbose
        # PPU: minimal path compiles the CUDA-derived source with nvrtc, same as CUDA.
        if is_ppu_target(target):
            # PPU: no RTC equivalent — compile the device source to .hgbin AOT
            # with hgcc (same command line as tilelang_callback_ppu_compile).
            self._compile_lib_ppu()
        elif is_cuda_target(target):
            if not is_nvrtc_available:
                raise ImportError(NVRTC_UNAVAILABLE_MESSAGE)
            from tilelang.env import CUDA_HOME, CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH

            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cu", delete=False)
            libpath = src.name.replace(".cu", ".cubin")

            project_root = osp.join(osp.dirname(__file__), "..", "..", "..", "..")
            if CUTLASS_INCLUDE_DIR is None:
                cutlass_path = osp.abspath(osp.join(project_root, "3rdparty/cutlass/include"))
            else:
                cutlass_path = CUTLASS_INCLUDE_DIR

            if TILELANG_TEMPLATE_PATH is None:
                tl_template_path = osp.abspath(osp.join(project_root, "src"))
            else:
                tl_template_path = TILELANG_TEMPLATE_PATH

            cuda_home = CUDA_HOME if CUDA_HOME else "/usr/local/cuda"

            cuda_include_paths = discover_cuda_include_paths(cuda_home)

            __CUDACC_VER_MAJOR__ = get_nvrtc_version()[0]
            options = [
                f"-I{tl_template_path}",
                f"-I{cutlass_path}",
                *(f"-I{include_path}" for include_path in cuda_include_paths),
                f"-D__CUDACC_VER_MAJOR__={__CUDACC_VER_MAJOR__}",
            ]

            # CUDA <13 keeps cuda::std at the legacy ``cuda/std`` path. CUDA 13
            # moved it under CCCL (``cccl/cuda/std``) which is already reachable
            # via the ``-I {arch_include}/cccl`` entry above as ``<cuda/std/*>``.
            # Adding ``-I .../cccl/cuda/std`` would also expose CCCL's private
            # ``__tuple_dir/structured_bindings.h`` at the include root, which
            # collides ODR-style with cutlass's own ``tuple_size``/``tuple_element``
            # forward declarations in ``cute/container/tuple.hpp`` under NVRTC
            # (cute uses variadic packs, cccl uses a single ``_Tp``).
            if __CUDACC_VER_MAJOR__ < 13:
                options += [
                    f"-I{include_path}/cuda/std"
                    for include_path in cuda_include_paths
                    if not include_path.endswith(osp.join("include", "cccl"))
                ]

            if self.compile_flags:
                options += [item for flag in self.compile_flags for item in flag.split() if item not in options]

            cubin_bytes = compile_cuda(self.lib_code, target_format="cubin", options=options, verbose=verbose)
            with open(libpath, "wb") as f:
                f.write(cubin_bytes)

            src.write(self.lib_code)
            src.flush()

            self.srcpath = src.name
            self.libpath = libpath
            self.pypath = src.name.replace(".cu", ".py")
            if self.host_func is None:
                raise RuntimeError("Host function is not set, please call update_host_func() first.")
            with open(self.pypath, "w") as f:
                f.write(self.host_func)
        else:
            raise ValueError(f"Unsupported target: {target}")

    def _compile_lib_ppu(self):
        """Compile PPU device source to .hgbin using hgcc (via tilelang.contrib.hgcc)."""
        from tilelang.contrib import hgcc
        from tilelang.transform import PassConfigKey

        cfg = self.pass_configs or {}
        verbose = bool(cfg.get(PassConfigKey.TL_ENABLE_PTXAS_VERBOSE_OUTPUT, False))

        options = []
        extra_flags = cfg.get(PassConfigKey.TL_DEVICE_COMPILE_FLAGS, None)
        if extra_flags:
            import shlex

            if isinstance(extra_flags, str):
                options += shlex.split(extra_flags)
            else:
                for flag in extra_flags:
                    if isinstance(flag, str):
                        options.extend(shlex.split(flag))
                    else:
                        options.append(str(flag))

        if self.compile_flags:
            options += [item for flag in self.compile_flags for item in flag.split() if item not in options]

        # Write source code to temp file (srcpath not set in PPU path)
        src = tempfile.NamedTemporaryFile(mode="w", suffix=".hg", delete=False)
        src.write(self.lib_code)
        src.close()
        self.srcpath = src.name

        binary = hgcc.compile_ppu(self.lib_code, target=self.target, options=options, verbose=verbose)

        # Write binary to .hgbin file
        self.libpath = self.srcpath.replace(".hg", ".hgbin")
        with open(self.libpath, "wb") as f:
            f.write(binary)
        # Write host function to .py file (needed by NVRTC adapter)
        self.pypath = self.srcpath.replace(".hg", ".py")
        if self.host_func is None:
            raise RuntimeError("Host function is not set, please call update_host_func() first.")
        with open(self.pypath, "w") as f:
            f.write(self.host_func)

    def _load_module_ppu(self, lib_path: str):
        """Load an hgcc-compiled .hgbin through the HGGC driver (ctypes)."""
        drv = _get_hg_driver()
        with open(lib_path, "rb") as f:
            data = f.read()
        # Keep the image buffer alive for the lifetime of the module.
        self._hgbin_buf = ctypes.create_string_buffer(data, len(data))
        module = ctypes.c_void_p()
        res = drv.hgModuleLoadData(ctypes.byref(module), ctypes.cast(self._hgbin_buf, ctypes.c_void_p))
        if res != 0 or not module:
            raise RuntimeError(f"hgModuleLoadData failed for {lib_path}: HGresult={res}")
        self.hgmod = module

    def get_ppu_function(self, name: str):
        """Return the HGfunction handle (ctypes.c_void_p) of a kernel in the loaded PPU module."""
        drv = _get_hg_driver()
        func = ctypes.c_void_p()
        res = drv.hgModuleGetFunction(ctypes.byref(func), self.hgmod, name.encode("utf-8"))
        if res != 0 or not func:
            raise RuntimeError(f"hgModuleGetFunction({name}) failed: HGresult={res}")
        return func

    def __del__(self):
        """Cleanup: unload CUDA library when object is destroyed.

        Critical for resource management - CUDA libraries consume GPU memory.
        Failure to unload is logged but not raised (destructor can't fail).

        Why explicit unload:
            Python GC doesn't know about GPU resources, must release manually.
        """
        hgmod = getattr(self, "hgmod", None)
        if hgmod:
            try:
                _get_hg_driver().hgModuleUnload(hgmod)
            except Exception:
                logger.warning(f"Failed to unload PPU module: {self.libpath}")
            self.hgmod = None
        if self.culib:
            result = cuda.cuLibraryUnload(self.culib)[0]
            if result != cuda.CUresult.CUDA_SUCCESS:
                logger.warning(f"Failed to unload library: {self.libpath}")
            self.culib = None
