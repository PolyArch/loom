#include "loom/SystemCompiler/TDCVerification.h"

namespace loom {

//===----------------------------------------------------------------------===//
// TDCVerificationResult helpers
//===----------------------------------------------------------------------===//

void TDCVerificationResult::addWarning(const std::string &msg) {
  diagnostics.push_back({TDCDiagnostic::Severity::Warning, msg});
}

void TDCVerificationResult::addError(const std::string &msg) {
  valid = false;
  diagnostics.push_back({TDCDiagnostic::Severity::Error, msg});
}

//===----------------------------------------------------------------------===//
// Edge spec verification
//===----------------------------------------------------------------------===//

TDCVerificationResult verifyEdgeSpec(const TDCEdgeSpec &spec) {
  TDCVerificationResult result;

  if (spec.producerKernel.empty()) {
    result.addError("TDCEdgeSpec: producerKernel is empty");
  }

  if (spec.consumerKernel.empty()) {
    result.addError("TDCEdgeSpec: consumerKernel is empty");
  }

  if (spec.dataTypeName.empty()) {
    result.addError("TDCEdgeSpec: dataTypeName is empty");
  }

  // If shape is set, verify it parses successfully.
  if (spec.shape.has_value() && !spec.shape->empty()) {
    auto dims = parseShapeExpr(*spec.shape);
    if (dims.empty()) {
      result.addWarning("TDCEdgeSpec: shape '" + *spec.shape +
                        "' parsed to zero dimensions");
    }
  }

  // Warn if throughput is set but empty.
  if (spec.throughput.has_value() && spec.throughput->empty()) {
    result.addWarning(
        "TDCEdgeSpec: throughput is set but empty for edge " +
        spec.producerKernel + "->" + spec.consumerKernel);
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Path spec verification
//===----------------------------------------------------------------------===//

TDCVerificationResult verifyPathSpec(const TDCPathSpec &spec) {
  TDCVerificationResult result;

  if (spec.startProducer.empty()) {
    result.addError("TDCPathSpec: startProducer is empty");
  }

  if (spec.startConsumer.empty()) {
    result.addError("TDCPathSpec: startConsumer is empty");
  }

  if (spec.endProducer.empty()) {
    result.addError("TDCPathSpec: endProducer is empty");
  }

  if (spec.endConsumer.empty()) {
    result.addError("TDCPathSpec: endConsumer is empty");
  }

  if (spec.latency.empty()) {
    result.addError("TDCPathSpec: latency expression is empty");
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Batch verification
//===----------------------------------------------------------------------===//

TDCVerificationResult
verifyContracts(const std::vector<TDCEdgeSpec> &edges,
                const std::vector<TDCPathSpec> &paths) {
  TDCVerificationResult merged;

  for (const auto &edge : edges) {
    auto r = verifyEdgeSpec(edge);
    if (!r.valid)
      merged.valid = false;
    merged.diagnostics.insert(merged.diagnostics.end(),
                              r.diagnostics.begin(), r.diagnostics.end());
  }

  for (const auto &path : paths) {
    auto r = verifyPathSpec(path);
    if (!r.valid)
      merged.valid = false;
    merged.diagnostics.insert(merged.diagnostics.end(),
                              r.diagnostics.begin(), r.diagnostics.end());
  }

  return merged;
}

} // namespace loom
