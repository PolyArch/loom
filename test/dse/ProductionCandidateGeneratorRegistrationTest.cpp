#include "DSE/CandidateGenerator.h"
#include "DSE/DataflowRewriteCandidateGenerator.h"
#include "DSE/FabricTemplateCandidateGenerator.h"
#include "DSE/FuReverseSynthesis.h"
#include "DSE/MappingCandidateGenerator.h"
#include "DSE/ModelParameterCalibrationAcquisition.h"
#include "DSE/ModelParameterTrainingCandidateGenerator.h"
#include "DSE/PortableSpatialCoreRtlCandidateGenerator.h"
#include "DSE/RootCompleteSpatialPnrCandidateGenerator.h"
#include "DSE/RootCompleteSystemPnrCandidateGenerator.h"
#include "DSE/RootCompleteTechMappingCandidateGenerator.h"
#include "DSE/SpatialMappingFeedbackCandidateGenerator.h"
#include "DSE/SpatialMicroarchitectureCandidateGenerator.h"
#include "DSE/SpatialTopologyCandidateGenerator.h"
#include "DSE/StructuredExecutionShapeCandidateGenerator.h"
#include "DSE/StructuredMemoryCommunicationCandidateGenerator.h"
#include "DSE/StructuredOwnershipCandidateGenerator.h"
#include "DSE/StructuredScheduleCandidateGenerator.h"
#include "DSE/StructuredSpecialMathAccuracyCandidateGenerator.h"
#include "DSE/SystemCompositionCandidateGenerator.h"

#include "llvm/Support/Error.h"

#include <cstdlib>
#include <iostream>

namespace {

using loom::dse::CandidateGeneratorDescriptor;
using loom::dse::CandidateGeneratorKind;

void requireSuccess(llvm::Error error) {
  if (!error)
    return;
  std::cerr << llvm::toString(std::move(error)) << '\n';
  std::exit(1);
}

void requireRegistered(CandidateGeneratorKind kind,
                       const CandidateGeneratorDescriptor &descriptor) {
  if (loom::dse::findCandidateGeneratorDescriptor(kind) == &descriptor)
    return;
  std::cerr << "candidate generator kind " << kind.ordinal()
            << " did not resolve to its production descriptor\n";
  std::exit(1);
}

} // namespace

int main() {
  requireSuccess(loom::dse::registerSpatialPnrCandidateGenerator());
  requireSuccess(loom::dse::registerStructuredOwnershipCandidateGenerator());
  requireSuccess(loom::dse::registerStructuredScheduleCandidateGenerator());
  requireSuccess(
      loom::dse::registerStructuredExecutionShapeCandidateGenerator());
  requireSuccess(loom::dse::registerDataflowRewriteCandidateGenerator());
  requireSuccess(
      loom::dse::registerStructuredMemoryCommunicationCandidateGenerator());
  requireSuccess(
      loom::dse::registerRootCompleteTechMappingCandidateGenerator());
  requireSuccess(
      loom::dse::registerApplicationGraphTechMappingCandidateGenerator());
  requireSuccess(loom::dse::registerRootCompleteSpatialPnrCandidateGenerator());
  requireSuccess(loom::dse::registerSpatialMappingFeedbackCandidateGenerator());
  requireSuccess(loom::dse::registerRootCompleteSystemPnrCandidateGenerator());
  requireSuccess(loom::dse::registerApplicationSystemPnrCandidateGenerator());
  requireSuccess(
      loom::dse::registerStructuredSpecialMathAccuracyCandidateGenerator());
  requireSuccess(loom::dse::registerFabricTemplateCandidateGenerator());
  requireSuccess(loom::dse::registerFuReverseSynthesisCandidateGenerator());
  requireSuccess(loom::dse::registerSpatialTopologyCandidateGenerator());
  requireSuccess(
      loom::dse::registerSpatialMicroarchitectureCandidateGenerator());
  requireSuccess(loom::dse::registerSystemCompositionCandidateGenerator());
  requireSuccess(
      loom::dse::registerPortableSpatialCoreRtlCandidateGenerator());
  requireSuccess(loom::dse::registerFpaGbdtTrainingCandidateGenerator());
  requireSuccess(
      loom::dse::registerSystemRuntimeGbdtTrainingCandidateGenerator());
  requireSuccess(
      loom::dse::registerModelParameterCalibrationPromotionAcquisitions());

  requireRegistered(CandidateGeneratorKind(0),
                    loom::dse::spatialPnrCandidateGeneratorDescriptor());
  requireRegistered(
      CandidateGeneratorKind(1),
      loom::dse::structuredOwnershipCandidateGeneratorDescriptor());
  requireRegistered(
      CandidateGeneratorKind(2),
      loom::dse::structuredScheduleCandidateGeneratorDescriptor());
  requireRegistered(
      CandidateGeneratorKind(3),
      loom::dse::structuredExecutionShapeCandidateGeneratorDescriptor());
  requireRegistered(CandidateGeneratorKind(4),
                    loom::dse::dataflowRewriteCandidateGeneratorDescriptor());
  requireRegistered(
      CandidateGeneratorKind(5),
      loom::dse::structuredMemoryCommunicationCandidateGeneratorDescriptor());
  requireRegistered(
      CandidateGeneratorKind(6),
      loom::dse::rootCompleteTechMappingCandidateGeneratorDescriptor());
  requireRegistered(
      CandidateGeneratorKind(7),
      loom::dse::rootCompleteSpatialPnrCandidateGeneratorDescriptor());
  requireRegistered(
      CandidateGeneratorKind(8),
      loom::dse::spatialMappingFeedbackCandidateGeneratorDescriptor());
  requireRegistered(
      CandidateGeneratorKind(9),
      loom::dse::rootCompleteSystemPnrCandidateGeneratorDescriptor());
  requireRegistered(
      CandidateGeneratorKind(10),
      loom::dse::structuredSpecialMathAccuracyCandidateGeneratorDescriptor());
  requireRegistered(CandidateGeneratorKind(12),
                    loom::dse::fabricTemplateCandidateGeneratorDescriptor());
  requireRegistered(CandidateGeneratorKind(13),
                    loom::dse::spatialTopologyCandidateGeneratorDescriptor());
  requireRegistered(
      CandidateGeneratorKind(14),
      loom::dse::spatialMicroarchitectureCandidateGeneratorDescriptor());
  requireRegistered(CandidateGeneratorKind(15),
                    loom::dse::systemCompositionCandidateGeneratorDescriptor());
  requireRegistered(CandidateGeneratorKind(16),
                    loom::dse::portableSpatialCoreRtlCandidateGeneratorDescriptor());
  requireRegistered(CandidateGeneratorKind(17),
                    loom::dse::fpaGbdtTrainingCandidateGeneratorDescriptor());
  requireRegistered(
      CandidateGeneratorKind(18),
      loom::dse::systemRuntimeGbdtTrainingCandidateGeneratorDescriptor());
  requireRegistered(
      CandidateGeneratorKind(21),
      loom::dse::applicationGraphTechMappingCandidateGeneratorDescriptor());
  requireRegistered(
      CandidateGeneratorKind(22),
      loom::dse::applicationSystemPnrCandidateGeneratorDescriptor());
  requireRegistered(
      CandidateGeneratorKind(23),
      loom::dse::fuReverseSynthesisCandidateGeneratorDescriptor());
}
