#ifndef LOOM_EVALUATION_PRODUCTIONREGISTRY_H
#define LOOM_EVALUATION_PRODUCTIONREGISTRY_H

#include "Evaluation/ModelDescriptor.h"

#include <cstdint>

namespace loom::evaluation {

enum class BuiltinEvaluationCase : std::uint32_t {
  StructuredProgramWithFabric = 0,
  CanonicalDataflowWithFabric = 1,
  StructuredProgramFunctionalComparison = 2,
  CanonicalDataflowSimulation = 3,
  FpaModelParameterCalibration = 4,
  HardwareImplementationPhysical = 5,
  SystemSimulation = 6,
  CgraSimulation = 7,
  SimulationExecutionComparison = 8,
  CanonicalDataflowSourceFunctionalComparison = 9,
  FabricHardwareAnalysis = 10,
  SystemRuntimeModelParameterCalibration = 11,
  MappedRtlSimulation = 12,
};

enum class BuiltinEvaluationModel : std::uint32_t {
  StructuredFabricLowConfidence = 2,
  CanonicalDataflowFabricLowConfidence = 3,
  StructuredProgramFunctional = 4,
  DfgSimulator = 5,
  FpaModelParameterCalibration = 6,
  StructuredFabricCalibratedFpa = 7,
  CanonicalDataflowFabricCalibratedFpa = 8,
  CgraSimulator = 9,
  SimulationExecutionComparison = 10,
  CanonicalDataflowSourceFunctional = 11,
  CadenceVoltusStaticRail = 12,
  FabricLowConfidence = 13,
  FabricCalibratedFpa = 14,
  SystemRuntimeModelParameterCalibration = 15,
  Gem5CgraSystemRuntimePredictor = 16,
  Gem5SystemDfg = 17,
  Gem5SystemCgra = 18,
  Gem5SystemRtl = 19,
  OpenRoadRoutedStaticFpa = 20,
  MappedRtlSimulator = 21,
};

constexpr EvaluationCaseKind
builtinEvaluationCaseKind(BuiltinEvaluationCase value) {
  return EvaluationCaseKind(static_cast<std::uint32_t>(value));
}

constexpr EvaluationModelKind
builtinEvaluationModelKind(BuiltinEvaluationModel value) {
  return EvaluationModelKind(static_cast<std::uint32_t>(value));
}

llvm::Error registerProductionEvaluationRegistry();

EvaluationCaseSignatureRef
builtinEvaluationCaseSignatureRef(BuiltinEvaluationCase evaluationCase);
llvm::Expected<EvaluationModelDescriptorRef>
builtinEvaluationModelDescriptorRef(BuiltinEvaluationModel model);

EvaluationCaseSignatureRef systemSimulationCaseSignatureRef();
EvaluationCaseSignatureRef fabricHardwareAnalysisCaseSignatureRef();
EvaluationCaseSignatureRef systemRuntimeCalibrationCaseSignatureRef();
EvaluationCaseSignatureRef mappedRtlSimulationCaseSignatureRef();

} // namespace loom::evaluation

#endif // LOOM_EVALUATION_PRODUCTIONREGISTRY_H
