#include "DSE/ProductionOwners.h"

#include "DSE/DataflowEvaluationAcquisition.h"
#include "DSE/DataflowRewriteCandidateGenerator.h"
#include "DSE/FabricTemplateCandidateGenerator.h"
#include "DSE/ModelParameterCalibrationAcquisition.h"
#include "DSE/ModelParameterTrainingCandidateGenerator.h"
#include "DSE/PortableSpatialCoreRtlCandidateGenerator.h"
#include "DSE/RootCompleteSpatialPnrCandidateGenerator.h"
#include "DSE/RootCompleteSystemPnrCandidateGenerator.h"
#include "DSE/RootCompleteTechMappingCandidateGenerator.h"
#include "DSE/SpatialMappingEvaluationAcquisition.h"
#include "DSE/SpatialMappingFeedbackCandidateGenerator.h"
#include "DSE/SpatialMicroarchitectureCandidateGenerator.h"
#include "DSE/SpatialTopologyCandidateGenerator.h"
#include "DSE/StructuredEvaluationAcquisition.h"
#include "DSE/StructuredExecutionShapeCandidateGenerator.h"
#include "DSE/StructuredMemoryCommunicationCandidateGenerator.h"
#include "DSE/StructuredOwnershipCandidateGenerator.h"
#include "DSE/StructuredScheduleCandidateGenerator.h"
#include "DSE/StructuredSpecialMathAccuracyCandidateGenerator.h"
#include "DSE/SystemCompositionCandidateGenerator.h"
#include "Evaluation/ProductionRegistry.h"

#include <array>

namespace loom::dse {

llvm::Error registerProductionDseOwners() {
  if (llvm::Error error =
          evaluation::registerProductionEvaluationRegistry())
    return error;
  const std::array registrations = {
      &registerStructuredOwnershipCandidateGenerator,
      &registerStructuredScheduleCandidateGenerator,
      &registerStructuredExecutionShapeCandidateGenerator,
      &registerStructuredSpecialMathAccuracyCandidateGenerator,
      &registerStructuredMemoryCommunicationCandidateGenerator,
      &registerDataflowRewriteCandidateGenerator,
      &registerSpatialPnrCandidateGenerator,
      &registerRootCompleteTechMappingCandidateGenerator,
      &registerApplicationGraphTechMappingCandidateGenerator,
      &registerRootCompleteSpatialPnrCandidateGenerator,
      &registerSpatialMappingFeedbackCandidateGenerator,
      &registerRootCompleteSystemPnrCandidateGenerator,
      &registerApplicationSystemPnrCandidateGenerator,
      &registerFabricTemplateCandidateGenerator,
      &registerSpatialTopologyCandidateGenerator,
      &registerSpatialMicroarchitectureCandidateGenerator,
      &registerSystemCompositionCandidateGenerator,
      &registerPortableSpatialCoreRtlCandidateGenerator,
      &registerFpaGbdtTrainingCandidateGenerator,
      &registerSystemRuntimeGbdtTrainingCandidateGenerator,
      &registerStructuredEvaluationPromotionAcquisition,
      &registerDataflowEvaluationPromotionAcquisition,
      &registerSpatialMappingEvaluationPromotionAcquisition,
      &registerModelParameterCalibrationPromotionAcquisitions};
  for (llvm::Error (*registration)() : registrations)
    if (llvm::Error error = registration())
      return error;
  return llvm::Error::success();
}

} // namespace loom::dse
