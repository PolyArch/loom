#ifndef LOOM_ADG_BUILTIN_H
#define LOOM_ADG_BUILTIN_H

#include "ADG/Builder.h"
#include "ADG/BuiltinDescriptor.h"
#include "Common/ArtifactStore.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <vector>

namespace loom::adg {

/// One builtin SpatialCore recipe expanded into an open public Builder root.
/// The caller may route additional typed resources before closing the root
/// with either outputs or a replacement result sequence.
struct BuiltinSpatialCoreExpansion final {
  SpatialCoreBuilder spatialCore;
  std::vector<SpatialValue> outputs;
};

/// Exact InstructionCore architecture shared by all builtin target presets.
llvm::Expected<loom::fabric::InstructionCoreArchitecturalContract>
getBuiltinInstructionCoreArchitecture();

/// Exact in-order InstructionCore realization used by builtin target presets.
/// System recipes that embed an independently authored SpatialCore reuse this
/// owner instead of copying its execution-unit and resource contracts.
llvm::Expected<loom::fabric::InstructionCoreMicroarchitecturalRealization>
getBuiltinInOrderInstructionCoreMicroarchitecture();

llvm::Expected<BuiltinTargetPreset> parseBuiltinTargetPreset(llvm::StringRef);

llvm::Expected<BuiltinSpatialCoreExpansion>
expandBuiltinSpatialCore(DesignBuilder &design, BuiltinTargetPreset preset);
llvm::Expected<BuiltinSpatialCoreExpansion>
expandBuiltinSpatialCore(DesignBuilder &design,
                         const BuiltinTargetScale &scale);

/// Expands the System recipe around an independently finalized SpatialCore.
/// The builtin hardware domain is complete, while the returned System remains
/// open for additional typed resources and domains before close().
llvm::Expected<SystemBuilder>
expandBuiltinSystem(DesignBuilder &design, BuiltinTargetPreset preset,
                    const loom::fabric::FinalizedFabricRoot &spatialCore);
llvm::Expected<SystemBuilder>
expandBuiltinSystem(DesignBuilder &design, const BuiltinTargetScale &scale,
                    const loom::fabric::FinalizedFabricRoot &spatialCore);

/// Expands and finalizes the selected descriptor through the same public ADG
/// Builder API available to external hardware authors. The returned design
/// contains one System root; its Module dependency is independently published
/// in the supplied ArtifactStore.
llvm::Expected<FinalizedFabricDesign>
buildBuiltinTarget(const loom::ArtifactStore &store,
                   BuiltinTargetPreset preset);
llvm::Expected<FinalizedFabricDesign>
buildBuiltinTarget(const loom::ArtifactStore &store,
                   const BuiltinTargetScale &scale);
llvm::Expected<FinalizedFabricDesign>
buildBuiltinTarget(const loom::ArtifactStore &store,
                   llvm::StringRef templateIdentity, std::uint32_t schemaMajor,
                   std::uint32_t schemaMinor, const BuiltinTargetScale &scale);

} // namespace loom::adg

#endif // LOOM_ADG_BUILTIN_H
