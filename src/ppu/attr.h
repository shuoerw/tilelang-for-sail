/*!
 * \file tl/ppu/attr.h
 * \brief PPU-specific IR annotations.
 */

#ifndef TVM_TL_PPU_ATTR_H_
#define TVM_TL_PPU_ATTR_H_

namespace tvm {
namespace tl {
namespace attr {

// Transient marker used by stage-0 AIU optimization passes. It is attached
// only to AIU loads lowered from synchronous T.copy operations.
static constexpr const char *kAIUSyncOpt = "aiu_sync_opt";

} // namespace attr
} // namespace tl
} // namespace tvm

#endif // TVM_TL_PPU_ATTR_H_
