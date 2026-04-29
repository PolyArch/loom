#ifndef LOOM_FABRIC_TECH_SYNTHESIZER_SYNTHESIZER_H
#define LOOM_FABRIC_TECH_SYNTHESIZER_SYNTHESIZER_H

// Abstract Synthesizer interface and factory used by
// `loom-generalize-subgraphs-to-fu` to dispatch the per-group synthesis
// strategy chosen by `SynthConfig.strategy`.
//
// Strategies (`anchor`, `mcs`, `incremental`, `incremental_random`) all
// implement `Synthesizer::run` over a `SynthInputs` value bundle and
// return a `SynthResult` carrying either the freshly built wrapper
// `func.func` (containing one detached `fabric.fu`) or one of the closed
// `SynthFailureReason` enum values.
//
// Spec source: `docs/spec-generalize-subgraphs-to-fu.md`, sections
// "Strategies" (the C++ interface block) and
// "Failure reasons (closed enumeration)".
//
// Threading: the factory is stateless and safe to call from any thread.
// Concrete strategies must build their candidate wrappers in scratch
// `MLIRContext`s (or in the caller-supplied context) without mutating
// the user's `ModuleOp`; the pass main thread splices results in.

#include "Common/SynthConfig.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/IR/FabricOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace loom::fabric::tech {

// Closed enumeration of reasons synthesis can fail. Mirrors the spec's
// "Failure reasons (closed enumeration)" list verbatim. Names (in this
// enum) are PascalCase; the on-IR `loom.synth_failed` attribute string
// must be the snake_case spec wording produced by `failureReasonString`.
enum class SynthFailureReason : uint8_t {
  None = 0,                  // success sentinel
  CrossShareGroup,           // "cross_share_group"
  TopologyMismatch,          // "topology_mismatch"
  FeedbackAlignConflict,     // "feedback_align_conflict"
  CoverageVerifyFailed,      // "coverage_verify_failed"
  Timeout,                   // "timeout"
  ResourceExhausted,         // "resource_exhausted"
  UnsupportedOp,             // "unsupported_op"
  InvalidInput,              // "invalid_input"
  VerifierFailed,            // "verifier_failed"
  SymbolConflict,            // "symbol_conflict"
  ConfigParseFailed,         // "config_parse_failed"
  NoLegalMaterialization,    // "no_legal_materialization"
};

// Inverse: snake_case spec string for the attribute / diagnostic.
//
// `None` is reported as the empty string (`""`) so callers that splat
// the result into the `loom.synth_failed` attribute on success paths
// produce no token. Every non-`None` enumerator round-trips through the
// spec wording verbatim.
::llvm::StringRef failureReasonString(SynthFailureReason);

// Coverage report produced by `CoverageVerifier::verify`. Lives in this
// header so `SynthResult` can embed one by value without forming a
// circular dependency with the verifier (which depends on the
// `Synthesizer` family at link time, not at type time).
struct CoverageReport {
  // For each input subgraph, the index of a materialized FU candidate
  // that matches it, or `std::nullopt` on miss.
  ::llvm::SmallVector<::std::optional<::std::size_t>, 8> matchIndex;

  // True iff every input subgraph found a materialized match. Vacuous
  // coverage (`matchIndex` empty) returns `true`: zero inputs are
  // trivially covered by any FU. The verifier never produces an empty
  // `matchIndex` for a non-empty input list, so this default surfaces
  // only for callers that build a `SynthResult` with no verifier run.
  bool allCovered() const;
};

// Inputs to one Synthesizer run. References borrow; ownership is not
// transferred. The synthesizer must not store these references past the
// `run` call.
struct SynthInputs {
  // Lexical group name (the value of `loom.synth_group`, or
  // `"default"` for the implicit group).
  ::llvm::StringRef groupName;
  // One entry per input subgraph in this group. The synthesizer may
  // reorder its internal handling but must produce a `CoverageReport`
  // whose `matchIndex` is keyed by this slice's order.
  ::llvm::ArrayRef<::dataflow::SubgraphOp> subgraphs;
  // Resolved synth config (already parsed; defaults applied).
  const ::loom::SynthConfig &config;
  // MLIR context used to build the wrapper. Concrete strategies may
  // additionally allocate scratch contexts for parallel per-restart
  // work; the returned wrapper, however, must be created in `context`
  // so the pass main thread can splice it without re-parsing.
  ::mlir::MLIRContext *context = nullptr;
};

// Outputs from one Synthesizer run.
struct SynthResult {
  // On success: ownership of a freshly built wrapper `func.func` that
  // contains exactly one `fabric.fu` (detached, caller inserts into
  // the module). Null on failure.
  ::mlir::OwningOpRef<::mlir::func::FuncOp> wrapper;
  // `None` on success; one of the closed enum values on failure.
  SynthFailureReason failureReason = SynthFailureReason::None;
  // CoverageVerifier output. Default-constructed when the verifier is
  // disabled (or for failure paths that never reached verification).
  CoverageReport coverage;
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

// Abstract base for one synthesis strategy. One instance handles one
// group; concrete strategies must be safe to construct and destroy
// from any thread, but `run` itself need not be re-entrant.
class Synthesizer {
public:
  virtual ~Synthesizer() = default;
  virtual SynthResult run(const SynthInputs &) = 0;
};

// Factory: looks up `SynthConfig.strategy` and constructs the right
// concrete subclass. For strategies that are not yet implemented, this
// returns a stub that immediately reports `TopologyMismatch` with a
// note explaining that the named strategy is a stub. Returns `nullptr`
// on an unknown strategy name (caller must propagate as
// `invalid_input`).
//
// `strategyName` is matched exactly (no canonicalization); the four
// known names are `anchor`, `mcs`, `incremental`, `incremental_random`.
::std::unique_ptr<Synthesizer>
makeSynthesizer(::llvm::StringRef strategyName,
                const ::loom::SynthConfig &);

} // namespace loom::fabric::tech

#endif // LOOM_FABRIC_TECH_SYNTHESIZER_SYNTHESIZER_H
