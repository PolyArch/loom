#include "Fabric/Tech/Synthesizer/Synthesizer.h"

#include "Fabric/Tech/Synthesizer/Anchor.h"
#include "Fabric/Tech/Synthesizer/Incremental.h"
#include "Fabric/Tech/Synthesizer/IncrementalRandom.h"
#include "Fabric/Tech/Synthesizer/MCS.h"

#include "llvm/Support/Compiler.h"

#include <memory>

namespace loom::fabric::tech {

//===----------------------------------------------------------------------===//
// CoverageReport
//===----------------------------------------------------------------------===//

bool CoverageReport::allCovered() const {
  // Vacuous coverage: zero inputs are trivially covered. Any non-empty
  // matchIndex requires every entry to carry a materialized index.
  for (const ::std::optional<::std::size_t> &slot : matchIndex)
    if (!slot.has_value())
      return false;
  return true;
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
  case SynthFailureReason::NoLegalMaterialization:
    return "no_legal_materialization";
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

::std::unique_ptr<Synthesizer>
makeSynthesizer(::llvm::StringRef strategyName,
                const ::loom::SynthConfig &cfg) {
  // Known strategy names per `SynthConfig.strategy` documentation.
  if (strategyName == "anchor")
    return std::make_unique<AnchorSynthesizer>(cfg);
  if (strategyName == "incremental")
    return std::make_unique<IncrementalSynthesizer>(cfg);
  if (strategyName == "incremental_random")
    return std::make_unique<IncrementalRandomSynthesizer>(cfg);
  if (strategyName == "mcs")
    return std::make_unique<MCSSynthesizer>(cfg);

  // Unknown name: caller is responsible for translating this null
  // return into an `invalid_input` diagnostic on the input function.
  return nullptr;
}

} // namespace loom::fabric::tech
