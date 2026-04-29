#include "Fabric/Tech/Synthesizer/Synthesizer.h"

#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>
#include <utility>

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
  case SynthFailureReason::CoverageVerifyFailed:
    return "coverage_verify_failed";
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
// Stub strategy.
//
// Used by `makeSynthesizer` for every known strategy until each strategy
// task lands its real implementation. Returns a `TopologyMismatch`
// failure with a single `notes` line explaining that the named strategy
// is a stub. The choice of `TopologyMismatch` matches the "BLOCKED"
// signal documented in the per-task spec for Synthesizer scaffolding.
//===----------------------------------------------------------------------===//

namespace {

class StubSynthesizer : public Synthesizer {
public:
  explicit StubSynthesizer(::llvm::StringRef name)
      : strategyName(name.str()) {}

  SynthResult run(const SynthInputs &) override {
    SynthResult r;
    r.failureReason = SynthFailureReason::TopologyMismatch;
    std::string note;
    {
      ::llvm::raw_string_ostream os(note);
      os << "strategy " << strategyName << " not yet implemented";
    }
    r.notes.push_back(std::move(note));
    return r;
  }

private:
  std::string strategyName;
};

} // namespace

//===----------------------------------------------------------------------===//
// Factory.
//===----------------------------------------------------------------------===//

::std::unique_ptr<Synthesizer>
makeSynthesizer(::llvm::StringRef strategyName,
                const ::loom::SynthConfig &) {
  // Known strategy names per `SynthConfig.strategy` documentation. Each
  // currently dispatches to a stub that reports TopologyMismatch with a
  // note; later tasks swap in the real strategy classes behind the
  // same factory entry point.
  if (strategyName == "anchor" || strategyName == "mcs" ||
      strategyName == "incremental" ||
      strategyName == "incremental_random")
    return std::make_unique<StubSynthesizer>(strategyName);

  // Unknown name: caller is responsible for translating this null
  // return into an `invalid_input` diagnostic on the input function.
  return nullptr;
}

} // namespace loom::fabric::tech
