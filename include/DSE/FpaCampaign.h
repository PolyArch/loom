#ifndef LOOM_DSE_FPACAMPAIGN_H
#define LOOM_DSE_FPACAMPAIGN_H

#include "Common/ExternalFileFingerprint.h"
#include "Config/ResolvedConfig.h"
#include "DSE/PlanValue.h"
#include "EDA/Adapters/OpenSource/OpenRoad.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

#include "llvm/Support/Error.h"

#include <string>
#include <vector>

namespace loom {
class ArtifactStore;
} // namespace loom

namespace loom::dse {

/// The offline physical-implementation request of the FPA ground-truth
/// campaign. Every System root is lowered to one portable SpatialCore RTL
/// implementation per AccCore occurrence; every exact RTL implementation root
/// is synthesized to a standard-cell gate netlist and routed by the pinned
/// OpenROAD provider on one ASIC target. The two stages are separate plans
/// because the synthesis slot admits exactly one exact implementation. The
/// technology files enter the plan only through their exact content
/// fingerprints; the machine-local paths stay in the local tool configuration.
struct FpaPhysicalImplementationRequest final {
  std::vector<ArtifactRootReference> systems;
  std::vector<ArtifactRootReference> rtlImplementations;
  platform::AsicTarget asicTarget;
  std::vector<std::string> technologyCornerKeys;
  std::string selectedTechnologyCornerKey;
  std::string yosysProviderBuild;
  std::string openRoadProviderBuild;
  eda::open_source::OpenRoadPlacementParameters placement;
  ExternalFileFingerprint technologyLef;
  ExternalFileFingerprint cellLef;
  ExternalFileFingerprint liberty;
};

struct FpaRtlStageOutputs final {
  ArtifactRootReference system;
  ArtifactRootReference configurationAbi;
  PlanOutputRef rtl;
};

struct FpaPhysicalStageOutputs final {
  ArtifactRootReference rtlImplementation;
  PlanOutputRef gateNetlist;
  PlanOutputRef routed;
};

struct FpaPhysicalImplementationPlan final {
  ResolvedConfig resolvedConfig;
  ArtifactRootReference implementationPlatform;
  platform::TechnologyCornerRef technologyCorner;
  std::vector<FpaRtlStageOutputs> rtlStages;
  std::vector<FpaPhysicalStageOutputs> physicalStages;
  std::vector<ArtifactRootReference> semanticInputs;
};

/// Authors the finite resolved DSE plan of the requested stages through the
/// production RTL, Yosys, and OpenROAD candidate generators. The
/// ImplementationPlatform and every packed ConfigurationABI are published into
/// the ArtifactStore before the plan is projected, so the plan carries only
/// exact roots.
llvm::Expected<FpaPhysicalImplementationPlan>
buildFpaPhysicalImplementationPlan(FpaPhysicalImplementationRequest request,
                                   const ResolvedConfig &baseConfig,
                                   const ArtifactStore &artifactStore);

} // namespace loom::dse

#endif // LOOM_DSE_FPACAMPAIGN_H
