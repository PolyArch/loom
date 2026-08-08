#ifndef LOOM_ADG_BUILTINDESCRIPTOR_H
#define LOOM_ADG_BUILTINDESCRIPTOR_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace loom::adg {

enum class BuiltinTargetPreset : std::uint8_t { Small, Default, Large };

struct BuiltinTargetScale final {
  std::uint32_t accCoreCount;
  std::uint32_t spatialPeCount;
  std::uint32_t temporalPeCount;
  std::uint32_t spatialMemoryCount;
  std::uint32_t temporalMemoryCount;
  std::uint32_t temporalResidentContexts;
  std::uint32_t gatewayCount;
  std::uint64_t memoryCapacityBytes;
};

struct BuiltinTargetDescriptor final {
  BuiltinTargetPreset preset;
  llvm::StringLiteral name;
  llvm::StringLiteral templateIdentity;
  std::uint32_t schemaMajor;
  std::uint32_t schemaMinor;
  BuiltinTargetScale scale;
};

inline constexpr BuiltinTargetDescriptor builtinSmallTarget{
    BuiltinTargetPreset::Small,
    "small",
    "loom.adg.builtin.small",
    1,
    1,
    {4, 12, 4, 1, 1, 2, 2, 64 * 1024}};

inline constexpr BuiltinTargetDescriptor builtinDefaultTarget{
    BuiltinTargetPreset::Default,
    "default",
    "loom.adg.builtin.default",
    1,
    1,
    {8, 27, 9, 2, 2, 4, 4, 256 * 1024}};

inline constexpr BuiltinTargetDescriptor builtinLargeTarget{
    BuiltinTargetPreset::Large,
    "large",
    "loom.adg.builtin.large",
    1,
    1,
    {16, 48, 16, 4, 4, 8, 8, 1024 * 1024}};

constexpr const BuiltinTargetDescriptor &
getBuiltinTargetDescriptor(BuiltinTargetPreset preset) {
  switch (preset) {
  case BuiltinTargetPreset::Small:
    return builtinSmallTarget;
  case BuiltinTargetPreset::Default:
    return builtinDefaultTarget;
  case BuiltinTargetPreset::Large:
    return builtinLargeTarget;
  }
  return builtinDefaultTarget;
}

} // namespace loom::adg

#endif // LOOM_ADG_BUILTINDESCRIPTOR_H
