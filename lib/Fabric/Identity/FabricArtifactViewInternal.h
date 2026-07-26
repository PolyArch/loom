#ifndef LOOM_LIB_FABRIC_IDENTITY_FABRICARTIFACTVIEWINTERNAL_H
#define LOOM_LIB_FABRIC_IDENTITY_FABRICARTIFACTVIEWINTERNAL_H

#include "Common/Artifact.h"
#include "Fabric/Identity/FabricFuCapabilityTemplate.h"
#include "Fabric/Identity/FabricRefImport.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::fabric::detail {

struct FabricNestedOwnerViewData {
  std::uint64_t transportEndpointCount = 0;
  std::vector<FabricMemoryEndpointRole> memoryEndpointRoles;
  std::vector<std::uint64_t> inventoryCounts;
  std::optional<::fabric::ResourceContract> resourceContract;
};

struct FabricFuNodeViewData {
  FabricFuNodeKind kind = FabricFuNodeKind::Op;
  FabricNestedOwnerViewData owner;
};

struct FabricEntityViewData {
  FabricEntityKind kind = FabricEntityKind::FabricModuleTemplate;
  FabricNestedOwnerViewData owner;
  std::vector<FabricFuNodeViewData> fuNodes;
  std::vector<FabricFuCapabilityTemplateRecord> fuCapabilityTemplates;
  std::vector<FabricNestedOwnerViewData> memoryOperationPorts;
  std::vector<FabricNestedOwnerViewData> instructionContexts;
  std::vector<FabricNestedOwnerViewData> transferPatterns;
  std::optional<FabricNestedOwnerViewData> spatialCore;
  std::optional<FabricNestedOwnerViewData> instructionCore;
  std::optional<FabricNestedOwnerViewData> localMemoryService;
  std::optional<FabricFuTemplateRef> fuTemplate;
  std::optional<FabricHardwareDomainKind> hardwareDomainKind;
};

struct FabricArtifactViewData {
  ArtifactIdentity identity;
  FabricRootKind rootKind = FabricRootKind::Module;
  std::vector<FabricEntityViewData> entities;
  std::vector<FabricPointConnectionPayload> pointConnections;
  std::vector<FabricPhysicalTraversalRef> admittedTraversals;
};

llvm::Expected<FabricArtifactView>
buildFabricArtifactView(FabricArtifactViewData data);

} // namespace loom::fabric::detail

#endif // LOOM_LIB_FABRIC_IDENTITY_FABRICARTIFACTVIEWINTERNAL_H
