#ifndef LOOM_DATAFLOW_IR_DATAFLOWSYNCRENDEZVOUS_H
#define LOOM_DATAFLOW_IR_DATAFLOWSYNCRENDEZVOUS_H

#include "mlir/IR/ValueRange.h"

#include <cstddef>

namespace mlir {
class Location;
class OpBuilder;
class Value;
} // namespace mlir

namespace dataflow {

/// Builds the canonical binary rendezvous tree over `inputs` and returns the
/// tree output carrying `carrierLeaf`. Every input remains an atomic
/// prerequisite even though only the selected carrier is externally live.
/// Inputs must be nonempty and carrierLeaf must name one input.
mlir::Value buildCanonicalSyncRendezvousTree(mlir::OpBuilder &builder,
                                             mlir::Location location,
                                             mlir::ValueRange inputs,
                                             std::size_t carrierLeaf);

} // namespace dataflow

#endif // LOOM_DATAFLOW_IR_DATAFLOWSYNCRENDEZVOUS_H
