#ifndef LOOM_ADG_BUILTIN_H
#define LOOM_ADG_BUILTIN_H

#include "ADG/Builder.h"
#include "Common/ArtifactStore.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

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

/// One builtin SpatialCore recipe expanded into an open public Builder root.
/// The caller may route additional typed resources before closing the root
/// with either outputs or a replacement result sequence.
struct BuiltinSpatialCoreExpansion final {
  SpatialCoreBuilder spatialCore;
  std::vector<SpatialValue> outputs;
};

const BuiltinTargetDescriptor &
getBuiltinTargetDescriptor(BuiltinTargetPreset preset);

llvm::Expected<BuiltinTargetPreset> parseBuiltinTargetPreset(llvm::StringRef);

llvm::Expected<BuiltinSpatialCoreExpansion>
expandBuiltinSpatialCore(DesignBuilder &design, BuiltinTargetPreset preset);

/// Expands the System recipe around an independently finalized SpatialCore.
/// The builtin hardware domain is complete, while the returned System remains
/// open for additional typed resources and domains before close().
llvm::Expected<SystemBuilder>
expandBuiltinSystem(DesignBuilder &design, BuiltinTargetPreset preset,
                    const loom::fabric::FinalizedFabricRoot &spatialCore);

/// Expands and finalizes the selected descriptor through the same public ADG
/// Builder API available to external hardware authors. The returned design
/// contains one System root; its Module dependency is independently published
/// in the supplied ArtifactStore.
llvm::Expected<FinalizedFabricDesign>
buildBuiltinTarget(const loom::ArtifactStore &store,
                   BuiltinTargetPreset preset);

} // namespace loom::adg

#endif // LOOM_ADG_BUILTIN_H
