//===-- FabricConstants.h - Centralized fabric constants --------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Centralized constants shared across the fabric infrastructure.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_HARDWARE_COMMON_FABRICCONSTANTS_H
#define LOOM_HARDWARE_COMMON_FABRICCONSTANTS_H

namespace loom {

/// Address bit-width for index-type ports in the fabric.
/// Inspired by RISC-V Sv57 (57-bit virtual address).
/// Intentionally distinct from 32-bit and 64-bit data widths so that
/// address computation networks are never accidentally merged with
/// 64-bit data networks.
inline constexpr unsigned ADDR_BIT_WIDTH = 57;

} // namespace loom

#endif // LOOM_HARDWARE_COMMON_FABRICCONSTANTS_H
