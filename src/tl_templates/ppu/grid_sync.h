#pragma once

#include <cooperative_groups.h>
#include <hggc/atomic>

namespace tl {

__device__ __forceinline__ void sync_grid() {
#if defined(_HCG_VER) && _HCG_VER == 1001 && defined(_HCG_ABI_VER) &&          \
    _HCG_ABI_VER == 1
  auto grid = cooperative_groups::this_grid();
  if (!grid.is_valid()) {
    __trap();
  }
  auto *workspace = cooperative_groups::_hcg_impl::get_grid_ws_ptr();

  __barrier_sync(0);
  if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
    const unsigned int blocks = gridDim.x * gridDim.y * gridDim.z;
    constexpr unsigned int phase_bit = 0x80000000u;
    // Each block arrives once. The aggregate increment flips the phase bit
    // and leaves the low bits unchanged, so no racing reset is necessary.
    const unsigned int increment = ((blockIdx.x | blockIdx.y | blockIdx.z) == 0)
                                       ? (phase_bit - (blocks - 1))
                                       : 1u;
    hggc::atomic_ref<unsigned int, hggc::thread_scope_device> counter(
        workspace->sync_flag);
    const unsigned int prior =
        counter.fetch_add(increment, hggc::memory_order_release);
    while (((prior ^ counter.load(hggc::memory_order_acquire)) & phase_bit) ==
           0) {
    }
  }
  __barrier_sync(0);
#else
  cooperative_groups::this_grid().sync();
#endif
}

} // namespace tl
