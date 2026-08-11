#ifndef LOOM_EVALUATION_MODELS_CALIBRATEDFPA_H
#define LOOM_EVALUATION_MODELS_CALIBRATEDFPA_H

#include "Evaluation/Request.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <utility>

namespace loom::evaluation::models {

llvm::Error registerCalibratedFpaModels();
llvm::Error registerFpaModelParameterCalibrationModel();

EvaluationModelDescriptorRef structuredFabricCalibratedFpaModelDescriptorRef();
EvaluationModelDescriptorRef
canonicalDataflowFabricCalibratedFpaModelDescriptorRef();
EvaluationModelDescriptorRef fabricCalibratedFpaModelDescriptorRef();

EvaluationCaseSignatureRef fpaModelParameterCalibrationCaseSignatureRef();
EvaluationModelDescriptorRef fpaModelParameterCalibrationModelDescriptorRef();
CaseSubjectRoleRef fpaModelParameterCalibrationBundleRole();
CaseSubjectRoleRef fpaModelParameterCalibrationEvidenceRole();

/// Exact symmetric-relative-error aggregation used by the calibration model.
/// Pairs are predicted and observed canonical metric values respectively.
llvm::Expected<DecimalValue> calculateFpaPredictionErrorQuantile(
    llvm::ArrayRef<std::pair<DecimalValue, DecimalValue>> samples,
    ExactRatio quantile);

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_CALIBRATEDFPA_H
