#ifndef LOOM_EVALUATION_METRICTEXT_H
#define LOOM_EVALUATION_METRICTEXT_H

#include "Evaluation/Metric.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

namespace loom::evaluation {

void writeMetricObservationValueJson(
    llvm::json::OStream &json, const MetricObservationValue &observation);
llvm::Expected<MetricObservationValue>
parseMetricObservationValueJson(const llvm::json::Object &object);

} // namespace loom::evaluation

#endif // LOOM_EVALUATION_METRICTEXT_H
