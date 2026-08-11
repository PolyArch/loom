#ifndef LOOM_EVALUATION_MODELS_PREDICTIONCALIBRATION_H
#define LOOM_EVALUATION_MODELS_PREDICTIONCALIBRATION_H

#include "Evaluation/ModelDescriptor.h"
#include "Evaluation/NumericValue.h"

#include "llvm/ADT/ArrayRef.h"

namespace loom::evaluation::models {

llvm::Error registerPredictionCalibrationProviders();

llvm::Expected<DecimalValue>
calculateSymmetricRelativePredictionError(DecimalValue predicted,
                                          DecimalValue observed);

llvm::Expected<DecimalValue>
selectNearestRankPredictionError(llvm::ArrayRef<DecimalValue> values,
                                 ExactRatio probability);

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_PREDICTIONCALIBRATION_H
