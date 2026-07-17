#ifndef LOOM_FABRIC_TECH_SYNTHESIZER_SYNTHESIZER_H
#define LOOM_FABRIC_TECH_SYNTHESIZER_SYNTHESIZER_H

// Canonical FU synthesis interface used by
// `loom-synthesize-configured-functions` for each configured-function group.
//
// `synthesize` dispatches `SynthConfig.strategy`, runs the internal producer,
// and applies verification, coverage, and capability measurement before a
// successful `SynthResult` can be returned. The result carries either the
// freshly built wrapper
// `fabric.module` (containing one detached `fabric.pe` whose body holds
// the inner `fabric.fu`) or one of the closed `SynthFailureReason` enum
// values.
//
// Spec source: `docs/spec-generalize-subgraphs-to-fu.md`, sections
// "Strategies" (the C++ interface block) and
// "Failure reasons (closed enumeration)".
//
// Threading: `synthesize` is safe to call from any worker thread. Internal
// producers build their candidate wrappers in the
// worker-local scratch `MLIRContext` provided via `SynthInputs.context`
// (never in the user's module context) and must not mutate the user's
// `ModuleOp`. The pass main thread re-homes each returned wrapper into
// the user's module context and splices it in serially. See
// `SynthInputs.context` and `SynthResult.wrapper` for the contract.

#include "Common/SynthConfig.h"
#include "Fabric/IR/ConfiguredFunction.h"
#include "Fabric/IR/FabricOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace loom::fabric::tech {

// Closed enumeration of reasons synthesis can fail. Mirrors the spec's
// "Failure reasons (closed enumeration)" list verbatim. Names (in this
// enum) are PascalCase; the on-IR `loom.synth_failed` attribute string
// must be the snake_case spec wording produced by `failureReasonString`.
enum class SynthFailureReason : uint8_t {
  None = 0,              // success sentinel
  CrossShareGroup,       // "cross_share_group"
  TopologyMismatch,      // "topology_mismatch"
  FeedbackAlignConflict, // "feedback_align_conflict"
  Timeout,               // "timeout"
  ResourceExhausted,     // "resource_exhausted"
  UnsupportedOp,         // "unsupported_op"
  InvalidInput,          // "invalid_input"
  VerifierFailed,        // "verifier_failed"
  SymbolConflict,        // "symbol_conflict"
  ConfigParseFailed,     // "config_parse_failed"
};

// Inverse: snake_case spec string for the attribute / diagnostic.
//
// `None` is reported as the empty string (`""`) so callers that splat
// the result into the `loom.synth_failed` attribute on success paths
// produce no token. Every non-`None` enumerator round-trips through the
// spec wording verbatim.
::llvm::StringRef failureReasonString(SynthFailureReason);

struct CoverageWitness {
  ::std::size_t encodingIndex = 0;
  ::llvm::SmallVector<unsigned, 8> actorToFabricOp;
  ::llvm::SmallVector<::std::pair<unsigned, unsigned>, 4> inputPorts;
  ::llvm::SmallVector<::std::pair<unsigned, unsigned>, 4> outputPorts;
};

struct CoverageReport {
  // Complete match witness keyed by input order. A present witness carries
  // the selected encoding, actor-to-fabric.op correspondence, and exact
  // boundary-port mapping needed by a structural realization.
  ::llvm::SmallVector<::std::optional<CoverageWitness>, 8> witnesses;

  // True iff every input function has a complete witness. An empty input set
  // is vacuously covered.
  bool allCovered() const;
};

struct CapabilityMetrics {
  ::std::size_t encodingCount = 0;
  ::std::size_t coveredEncodingCount = 0;
  ::std::size_t extraCapabilityCount = 0;
};

CapabilityMetrics measureCapability(::fabric::FuOp fu,
                                    const CoverageReport &coverage);

struct SynthCandidateScore {
  double hardwareCost = 0.0;
  CapabilityMetrics capability;
  ::std::size_t deterministicOrder = 0;
};

// Lower hardware cost wins. Equivalent primary cost prefers less extra
// capability, then fewer total encodings, then stable producer order.
bool preferSynthCandidate(const SynthCandidateScore &candidate,
                          const SynthCandidateScore &currentBest);

// Inputs to one synthesis call. References borrow; ownership is not
// transferred past the call.
struct SynthInputs {
  // Lexical group name (the value of `loom.synth_group`, or
  // `"default"` for the implicit group).
  ::llvm::StringRef groupName;
  // One canonical software function per input in this group. The synthesizer
  // may reorder its internal handling but must produce a `CoverageReport` whose
  // witnesses are keyed by this slice's order.
  ::llvm::ArrayRef<::fabric::ConfiguredFunction> functions;
  // Scratch MLIR context for this worker. Per the spec rule "MLIR
  // mutation is never parallel", the pass constructs a fresh
  // thread-local `MLIRContext` for each worker before invoking `synthesize`,
  // and the internal producer must build its candidate
  // wrapper here -- never in the user's module context. Concrete
  // strategies may additionally allocate further scratch contexts for
  // parallel per-restart work. The pass's main thread is responsible
  // for re-homing the returned wrapper into the user's module context
  // (see `SynthResult::wrapper`).
  ::mlir::MLIRContext *context = nullptr;
};

// Outputs from one Synthesizer run.
struct SynthResult {
  // On success: ownership of a freshly built wrapper `fabric.module`
  // that contains exactly one `fabric.pe` whose body holds the inner
  // `fabric.fu` (detached, caller inserts into the module). The
  // wrapper is allocated in `SynthInputs.context` (the worker's
  // scratch context). The pass's main-thread splice loop is
  // responsible for cloning it into the user's module context before
  // insertion. Null on failure.
  ::mlir::OwningOpRef<::fabric::ModuleOp> wrapper;
  // `None` on success; one of the closed enum values on failure.
  SynthFailureReason failureReason = SynthFailureReason::None;
  // CoverageVerifier output. Default-constructed when the verifier is
  // disabled (or for failure paths that never reached verification).
  CoverageReport coverage;
  // Explicit capability metrics derived from the verified encoding set and
  // distinct input coverage witnesses.
  CapabilityMetrics capability;
  // Diagnostics emitted during synthesis (informational; not an error
  // log). Strategies should keep these short and machine-readable so
  // lit tests can assert against them.
  ::llvm::SmallVector<::std::string, 4> notes;

  // Convenience: `true` iff the wrapper is set and the failure reason
  // is `None`. The two fields can disagree only for buggy strategies;
  // the pass treats them via this single accessor.
  bool success() const {
    return wrapper && failureReason == SynthFailureReason::None;
  }
};

// The sole public accepted-result entrypoint dispatches the selected strategy,
// then applies wrapper verification, explicit encoding coverage, and
// capability measurement exactly once before returning a successful result.
SynthResult synthesize(const ::loom::SynthConfig &cfg,
                       const SynthInputs &inputs);

} // namespace loom::fabric::tech

#endif // LOOM_FABRIC_TECH_SYNTHESIZER_SYNTHESIZER_H
