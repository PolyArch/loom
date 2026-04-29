#ifndef LOOM_FABRIC_TECH_SYNTHESIZER_HW_PARAMS_H
#define LOOM_FABRIC_TECH_SYNTHESIZER_HW_PARAMS_H

// Shared helper that builds the `hw_params` ArrayAttr for a synthesized
// `fabric.op` from the union of attributes observed on the source ops at
// a merged position.
//
// Per spec section "hw_params policy": `SubgraphEnumerator` only fans
// out attribute axes that appear in `hw_params`. Emitting `[{}]` for an
// op that has a configurable axis (e.g. `arith.cmpi`'s `predicate`)
// would prevent the synthesized FU from enumerating any matching
// candidate, so coverage verification would fail. The helper covers:
//
//   * `arith.cmpi` / `arith.cmpf` - `predicate`
//   * `dataflow.stream`           - `step_op`, `cont_cond`
//   * `dataflow.constant`         - `const_hex_value`
//   * `dataflow.sync` / `dataflow.mux` / `dataflow.demux`
//                                 - `bitmask`
//
// For every other op kind (e.g. `arith.addi`) the helper returns the
// canonical `[{}]` value.
//
// Determinism:
//   * Dictionary keys are sorted lexically by `DictionaryAttr::get`.
//   * Each emitted array value is sorted lexically and deduplicated.

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace loom::fabric::tech {

// Build the observed-value union `hw_params` for the merged op named
// `opName` whose source-side peers are `peers`. Returns a length-1
// `ArrayAttr` wrapping a `DictionaryAttr`. When the op kind has no
// configurable axis (or the peers carry no recognized attribute), the
// dictionary is empty (`[{}]`).
::mlir::ArrayAttr
buildHwParamsUnion(::mlir::MLIRContext *ctx, ::llvm::StringRef opName,
                   ::llvm::ArrayRef<::mlir::Operation *> peers);

} // namespace loom::fabric::tech

#endif // LOOM_FABRIC_TECH_SYNTHESIZER_HW_PARAMS_H
