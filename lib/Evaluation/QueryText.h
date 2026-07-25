#ifndef LOOM_LIB_EVALUATION_QUERY_TEXT_H
#define LOOM_LIB_EVALUATION_QUERY_TEXT_H

#include "Evaluation/Finding.h"
#include "Evaluation/Metric.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

namespace loom::evaluation::detail {

void writeMetricQueryPayload(llvm::json::OStream &json,
                             const MetricQuery &query);
llvm::Expected<MetricQuery> parseMetricQueryPayload(
    const llvm::json::Object &object, llvm::StringRef context,
    llvm::ArrayRef<llvm::StringRef> envelopeFields = {});

void writeFindingQueryPayload(llvm::json::OStream &json,
                              const FindingQuery &query);
llvm::Expected<FindingQuery> parseFindingQueryPayload(
    const llvm::json::Object &object, llvm::StringRef context,
    llvm::ArrayRef<llvm::StringRef> envelopeFields = {});

} // namespace loom::evaluation::detail

#endif // LOOM_LIB_EVALUATION_QUERY_TEXT_H
