#ifndef LOOM_EVALUATION_MODELS_PHYSICALRAILANALYSIS_H
#define LOOM_EVALUATION_MODELS_PHYSICALRAILANALYSIS_H

#include "Evaluation/Models/PhysicalRailAnalysisConfig.h"
#include "Evaluation/Request.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace loom {
class ArtifactStore;
class BlobStore;

namespace hardware {
class ExternalImplementationContractCatalog;
}
} // namespace loom

namespace loom::evaluation::models {

inline constexpr llvm::StringLiteral
    cadenceVoltusRailImplementationSemanticIdentity =
        "loom.eda.cadence.voltus.rail@1";

enum class RailAnalysisMethod : std::uint8_t {
  Static,
};

enum class RailActivityBasis : std::uint8_t {
  ExplicitAssumption,
};

enum class RailNetworkCoverage : std::uint8_t {
  CompleteAnalyzedNetwork,
};

struct RailAnalysisModelConfig final {
  RailAnalysisMethod method;
  RailActivityBasis activityBasis;
  RailNetworkCoverage networkCoverage;
  UncertaintyKind uncertainty;

  friend bool operator==(RailAnalysisModelConfig lhs,
                         RailAnalysisModelConfig rhs) {
    return lhs.method == rhs.method && lhs.activityBasis == rhs.activityBasis &&
           lhs.networkCoverage == rhs.networkCoverage &&
           lhs.uncertainty == rhs.uncertainty;
  }
  friend bool operator!=(RailAnalysisModelConfig lhs,
                         RailAnalysisModelConfig rhs) {
    return !(lhs == rhs);
  }
};

struct ExplicitRailActivityBinding final {
  SubjectTargetRef target;
  ExplicitAssumptionSource assumption;
};

/// The unique provider input projected from model kind 12 and its exact
/// validated Request. Fixed model facts remain in the config-view contract;
/// operating conditions remain in the Request and are returned without a
/// second persistent encoding.
struct CompleteRailAnalysisConfiguration final {
  RailAnalysisModelConfig model;
  CadenceVoltusStaticRailProviderBinding providerBinding;
  ProcessCornerCondition processCorner;
  SupplyVoltageCondition supplyVoltage;
  TemperatureCondition temperature;
  RequiredClockPeriodCondition clockPeriod;
  ExplicitRailActivityBinding activity;
};

llvm::Error registerCadenceVoltusStaticRailModel();

EvaluationCaseSignatureRef hardwareImplementationPhysicalCaseSignatureRef();
EvaluationModelDescriptorRef cadenceVoltusStaticRailModelDescriptorRef();
CaseSubjectRoleRef hardwareImplementationPhysicalSubjectRole();
llvm::ArrayRef<ConditionApplicabilityPattern>
hardwareImplementationPhysicalBaseConditionPatterns();

llvm::Expected<CaseArtifactResolution>
resolveHardwareImplementationPhysicalCase(
    const ArtifactRootReference &hardwareImplementation,
    const hardware::ExternalImplementationContractCatalog &externalContracts,
    const ArtifactStore &artifactStore, const BlobStore &blobStore);

const RailAnalysisModelConfig &staticExplicitRailAnalysisModelConfig();

llvm::Expected<CompleteRailAnalysisConfiguration>
projectCompleteRailAnalysisConfiguration(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore, const BlobStore &blobStore);

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_PHYSICALRAILANALYSIS_H
