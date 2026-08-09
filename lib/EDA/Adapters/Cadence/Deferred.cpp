#include "EDA/Adapters/Cadence/Deferred.h"

namespace loom::eda::cadence {
namespace {

llvm::StringRef stageName(CadenceDeferredStage stage) {
  switch (stage) {
  case CadenceDeferredStage::XceliumFunctionalEvaluation:
    return "xcelium_functional_evaluation";
  case CadenceDeferredStage::InnovusAsicPhysical:
    return "innovus_asic_physical";
  case CadenceDeferredStage::JoulesPowerEvaluation:
    return "joules_power_evaluation";
  case CadenceDeferredStage::TempusTimingEvaluation:
    return "tempus_timing_evaluation";
  case CadenceDeferredStage::VoltusRailEvaluation:
    return "voltus_rail_evaluation";
  }
  return "unknown_cadence_stage";
}

llvm::StringRef boundaryName(CadenceDeferredBoundary boundary) {
  switch (boundary) {
  case CadenceDeferredBoundary::Prepare:
    return "prepare";
  case CadenceDeferredBoundary::Parse:
    return "parse";
  case CadenceDeferredBoundary::StrictImport:
    return "strict_import";
  }
  return "unknown_boundary";
}

std::string missingOwner(CadenceDeferredStage stage) {
  switch (stage) {
  case CadenceDeferredStage::XceliumFunctionalEvaluation:
    return "exact non-Spatial SimulationExecution and functional Evaluation "
           "publication forms";
  case CadenceDeferredStage::InnovusAsicPhysical:
    return "an admitted AsicPhysical representation format descriptor";
  case CadenceDeferredStage::JoulesPowerEvaluation:
    return "a normalized power Evaluation publication form";
  case CadenceDeferredStage::TempusTimingEvaluation:
    return "a normalized timing Evaluation publication form";
  case CadenceDeferredStage::VoltusRailEvaluation:
    return "a normalized rail-integrity Evaluation publication form";
  }
  return "a registered shared publication owner";
}

llvm::Error unsupported(CadenceDeferredStage stage,
                        CadenceDeferredBoundary boundary) {
  return llvm::make_error<CadenceStageUnsupportedError>(stage, boundary,
                                                        missingOwner(stage));
}

} // namespace

char CadenceStageUnsupportedError::ID = 0;

void CadenceStageUnsupportedError::log(llvm::raw_ostream &stream) const {
  stream << "cadence_stage_unsupported[" << stageName(stage_) << ":"
         << boundaryName(boundary_) << "]: missing " << missingOwner_;
}

std::error_code CadenceStageUnsupportedError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Expected<external_tool::PreparedExternalToolInvocation>
prepareDeferredCadenceStage(CadenceDeferredStage stage) {
  return unsupported(stage, CadenceDeferredBoundary::Prepare);
}

llvm::Error parseDeferredCadenceStage(CadenceDeferredStage stage,
                                      llvm::StringRef) {
  return unsupported(stage, CadenceDeferredBoundary::Parse);
}

llvm::Error importDeferredCadenceStage(
    CadenceDeferredStage stage,
    const external_tool::PreparedExternalToolInvocation &) {
  return unsupported(stage, CadenceDeferredBoundary::StrictImport);
}

} // namespace loom::eda::cadence
