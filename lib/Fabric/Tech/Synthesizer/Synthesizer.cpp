#include "Fabric/Tech/Synthesizer/Synthesizer.h"

#include "Fabric/IR/ConfiguredFunction.h"
#include "Fabric/Tech/Synthesizer/Anchor.h"
#include "Fabric/Tech/Synthesizer/CoverageVerifier.h"

#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

namespace loom::fabric::tech {

namespace {

::fabric::FuOp findInnerFu(::fabric::ModuleOp wrapper) {
  if (!wrapper)
    return nullptr;
  ::fabric::FuOp found;
  wrapper.walk([&](::fabric::FuOp fu) {
    if (!found)
      found = fu;
  });
  return found;
}

void enforceCanonicalAcceptance(
    SynthResult &result, ::llvm::ArrayRef<::fabric::ConfiguredFunction> inputs,
    const ::loom::SynthConfig &cfg) {
  if (!result.success())
    return;

  ::fabric::FuOp fu = findInnerFu(result.wrapper.get());
  if (!fu || ::mlir::failed(::mlir::verify(result.wrapper.get()))) {
    result.wrapper = nullptr;
    result.failureReason = SynthFailureReason::VerifierFailed;
    result.notes.push_back(
        "canonical synthesis gate: wrapper or FU verification failed");
    return;
  }

  CoverageVerifier verifier(cfg);
  result.coverage = verifier.verify(fu, inputs);
  if (!result.coverage.allCovered()) {
    result.wrapper = nullptr;
    result.failureReason = SynthFailureReason::VerifierFailed;
    result.notes.push_back(
        "canonical synthesis gate: explicit encodings do not cover every "
        "input function");
    return;
  }
  result.capability = measureCapability(fu, result.coverage);
}

} // namespace

//===----------------------------------------------------------------------===//
// CoverageReport
//===----------------------------------------------------------------------===//

bool CoverageReport::allCovered() const {
  for (const ::std::optional<CoverageWitness> &witness : witnesses)
    if (!witness.has_value())
      return false;
  return true;
}

CapabilityMetrics measureCapability(::fabric::FuOp fu,
                                    const CoverageReport &coverage) {
  CapabilityMetrics metrics;
  metrics.encodingCount = ::fabric::getValidSemanticEncodingCount(fu);
  ::llvm::DenseSet<::std::size_t> covered;
  for (const auto &witness : coverage.witnesses) {
    if (witness && witness->encodingIndex < metrics.encodingCount)
      covered.insert(witness->encodingIndex);
  }
  metrics.coveredEncodingCount = covered.size();
  metrics.extraCapabilityCount =
      metrics.encodingCount - metrics.coveredEncodingCount;
  return metrics;
}

bool preferSynthCandidate(const SynthCandidateScore &candidate,
                          const SynthCandidateScore &currentBest) {
  if (candidate.hardwareCost != currentBest.hardwareCost)
    return candidate.hardwareCost < currentBest.hardwareCost;
  if (candidate.capability.extraCapabilityCount !=
      currentBest.capability.extraCapabilityCount)
    return candidate.capability.extraCapabilityCount <
           currentBest.capability.extraCapabilityCount;
  if (candidate.capability.encodingCount !=
      currentBest.capability.encodingCount)
    return candidate.capability.encodingCount <
           currentBest.capability.encodingCount;
  return candidate.deterministicOrder < currentBest.deterministicOrder;
}

//===----------------------------------------------------------------------===//
// SynthFailureReason -> snake_case spec string.
//===----------------------------------------------------------------------===//

::llvm::StringRef failureReasonString(SynthFailureReason r) {
  switch (r) {
  case SynthFailureReason::None:
    return ::llvm::StringRef();
  case SynthFailureReason::CrossShareGroup:
    return "cross_share_group";
  case SynthFailureReason::TopologyMismatch:
    return "topology_mismatch";
  case SynthFailureReason::FeedbackAlignConflict:
    return "feedback_align_conflict";
  case SynthFailureReason::Timeout:
    return "timeout";
  case SynthFailureReason::ResourceExhausted:
    return "resource_exhausted";
  case SynthFailureReason::UnsupportedOp:
    return "unsupported_op";
  case SynthFailureReason::InvalidInput:
    return "invalid_input";
  case SynthFailureReason::VerifierFailed:
    return "verifier_failed";
  case SynthFailureReason::SymbolConflict:
    return "symbol_conflict";
  case SynthFailureReason::ConfigParseFailed:
    return "config_parse_failed";
  }
  // The switch above is exhaustive over the closed enum; this point is
  // unreachable. The builtin keeps optimizers from emitting a default
  // path that would mask a future enum addition (which would also fail
  // to compile under -Wswitch).
  LLVM_BUILTIN_UNREACHABLE;
}

//===----------------------------------------------------------------------===//
// Factory.
//===----------------------------------------------------------------------===//

::std::unique_ptr<Synthesizer> makeSynthesizer(::llvm::StringRef strategyName,
                                               const ::loom::SynthConfig &cfg) {
  if (strategyName == "anchor")
    return std::make_unique<AnchorSynthesizer>(cfg);
  return nullptr;
}

SynthResult synthesize(const ::loom::SynthConfig &cfg,
                       const SynthInputs &inputs) {
  auto strategy = makeSynthesizer(cfg.strategy, cfg);
  if (!strategy) {
    SynthResult result;
    result.failureReason = SynthFailureReason::InvalidInput;
    std::string note;
    ::llvm::raw_string_ostream os(note);
    os << "unknown strategy '" << cfg.strategy << "'";
    os.flush();
    result.notes.push_back(std::move(note));
    return result;
  }

  SynthResult result = strategy->run(inputs);
  enforceCanonicalAcceptance(result, inputs.functions, cfg);
  return result;
}

} // namespace loom::fabric::tech
