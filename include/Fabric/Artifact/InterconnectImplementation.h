#ifndef LOOM_FABRIC_ARTIFACT_INTERCONNECTIMPLEMENTATION_H
#define LOOM_FABRIC_ARTIFACT_INTERCONNECTIMPLEMENTATION_H

#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricEnums.h"

#include <cstdint>
#include <utility>

namespace loom::fabric {

/// Counts the protocol objects and architecture refinements in one strictly
/// imported implementation. The counts are an inspection projection; the
/// canonical implementation body remains the semantic owner.
struct InterconnectImplementationSummary final {
  ::fabric::InterconnectProtocolSchema protocol =
      ::fabric::InterconnectProtocolSchema::Gem5EventTransportV1;
  std::uint64_t endpointCount = 0;
  std::uint64_t resourceStateCount = 0;
  std::uint64_t transferPatternCount = 0;
  std::uint64_t configurationFieldCount = 0;
  std::uint64_t refinementCount = 0;
};

/// Typed authoring view for one concrete protocol provider. The builder owns
/// no shadow graph: finalization delegates to the registered provider and
/// returns the ordinary strictly imported Fabric artifact.
class InterconnectImplementationBuilder final {
public:
  static llvm::Expected<InterconnectImplementationBuilder>
  create(const FinalizedFabricRoot &refinedSystem,
         const ArtifactStore &store);

  InterconnectImplementationBuilder(
      const InterconnectImplementationBuilder &) = delete;
  InterconnectImplementationBuilder &operator=(
      const InterconnectImplementationBuilder &) = delete;
  InterconnectImplementationBuilder(
      InterconnectImplementationBuilder &&) noexcept = default;
  InterconnectImplementationBuilder &operator=(
      InterconnectImplementationBuilder &&) noexcept = default;

  /// Selects the closed protocol provider. Unsupported enum values are
  /// rejected before any artifact is published.
  llvm::Error setProtocolSchema(
      ::fabric::InterconnectProtocolSchema protocol);
  ::fabric::InterconnectProtocolSchema protocolSchema() const {
    return protocol_;
  }

  llvm::Expected<FinalizedFabricRoot> finalize() &&;

private:
  InterconnectImplementationBuilder(ArtifactRootReference refinedSystem,
                                    const ArtifactStore *store)
      : refinedSystem_(std::move(refinedSystem)), store_(store) {}

  ArtifactRootReference refinedSystem_;
  const ArtifactStore *store_ = nullptr;
  ::fabric::InterconnectProtocolSchema protocol_ =
      ::fabric::InterconnectProtocolSchema::Gem5EventTransportV1;
};

/// Materializes the complete event-transport implementation used by the gem5
/// System provider. The exact System remains the semantic architecture owner;
/// this sibling root owns only protocol-local objects and their total typed
/// refinement relation.
llvm::Expected<FinalizedFabricRoot>
finalizeGem5EventInterconnectImplementation(
    const ArtifactRootReference &refinedSystem, const ArtifactStore &store);

/// Returns the closed protocol schema selected by one strictly imported
/// InterconnectImplementation root. Other Fabric root kinds are rejected.
llvm::Expected<::fabric::InterconnectProtocolSchema>
interconnectProtocolSchema(const FinalizedFabricRoot &implementation);

llvm::Expected<InterconnectImplementationSummary>
inspectInterconnectImplementation(const FinalizedFabricRoot &implementation);

} // namespace loom::fabric

#endif // LOOM_FABRIC_ARTIFACT_INTERCONNECTIMPLEMENTATION_H
