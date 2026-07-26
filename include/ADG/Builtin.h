#ifndef LOOM_ADG_BUILTIN_H
#define LOOM_ADG_BUILTIN_H

#include "ADG/Builder.h"
#include "Common/ArtifactStore.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

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
  llvm::StringRef name;
  llvm::StringRef templateIdentity;
  std::uint32_t schemaMajor;
  std::uint32_t schemaMinor;
  BuiltinTargetScale scale;
};

const BuiltinTargetDescriptor &
getBuiltinTargetDescriptor(BuiltinTargetPreset preset);

llvm::Expected<BuiltinTargetPreset> parseBuiltinTargetPreset(llvm::StringRef);

/// Expands and finalizes the selected descriptor through the same public ADG
/// Builder API available to external hardware authors. The returned design
/// contains one System root; its Module dependency is independently published
/// in the supplied ArtifactStore.
llvm::Expected<FinalizedFabricDesign>
buildBuiltinTarget(const loom::ArtifactStore &store,
                   BuiltinTargetPreset preset);

} // namespace loom::adg

#endif // LOOM_ADG_BUILTIN_H
