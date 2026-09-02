#ifndef LOOM_ADG_SPECIALMATHCAPABILITYPROFILE_H
#define LOOM_ADG_SPECIALMATHCAPABILITYPROFILE_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace loom::adg {

/// Closed builtin special-math capability recipes. The profile selects only
/// the elementary-math inventory; scalar integer and floating divide/remainder
/// resources are invariant parts of SpecialMathFu.
enum class BuiltinSpecialMathCapabilityProfile : std::uint8_t {
  /// Every catalog floating format with strict IEEE behavior and a
  /// correctly-rounded guarantee for every elementary-math family.
  FullCatalog,
  /// Exact intersection of the production RTL provider format/behavior table.
  PortableProviderClosed,
};

constexpr bool isValidBuiltinSpecialMathCapabilityProfile(
    BuiltinSpecialMathCapabilityProfile profile) {
  return profile == BuiltinSpecialMathCapabilityProfile::FullCatalog ||
         profile ==
             BuiltinSpecialMathCapabilityProfile::PortableProviderClosed;
}

inline llvm::StringRef
builtinSpecialMathCapabilityProfileSpelling(
    BuiltinSpecialMathCapabilityProfile profile) {
  switch (profile) {
  case BuiltinSpecialMathCapabilityProfile::FullCatalog:
    return llvm::StringRef("full_catalog");
  case BuiltinSpecialMathCapabilityProfile::PortableProviderClosed:
    return llvm::StringRef("portable_provider_closed");
  }
  return {};
}

inline std::optional<BuiltinSpecialMathCapabilityProfile>
parseBuiltinSpecialMathCapabilityProfile(llvm::StringRef spelling) {
  if (spelling == "full_catalog")
    return BuiltinSpecialMathCapabilityProfile::FullCatalog;
  if (spelling == "portable_provider_closed")
    return BuiltinSpecialMathCapabilityProfile::PortableProviderClosed;
  return std::nullopt;
}

} // namespace loom::adg

#endif // LOOM_ADG_SPECIALMATHCAPABILITYPROFILE_H
