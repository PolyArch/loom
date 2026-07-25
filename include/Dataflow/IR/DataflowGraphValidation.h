#ifndef DATAFLOW_IR_DATAFLOWGRAPHVALIDATION_H
#define DATAFLOW_IR_DATAFLOWGRAPHVALIDATION_H

#include "Dataflow/IR/DataflowOps.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <optional>

namespace dataflow {

llvm::Error validateFinalizedGraph(GraphOp graph);
llvm::Error validateFinalizedProgram(::mlir::ModuleOp module);

/// One producer or consumer binding of a logical channel under a single root
/// thread launch. `site` is the `dataflow.channel.send`/`receive` operation or
/// the `dataflow.graph.launch` operation; `streamOrdinal` and `sourceMap` are
/// set only for a graph stream binding (the map only for a stream input).
struct ChannelEndpointBinding {
  ThreadLaunchOp rootLaunch;
  ThreadOp thread;
  unsigned threadArgumentOrdinal = 0;
  ::mlir::Operation *site = nullptr;
  std::optional<unsigned> streamOrdinal;
  std::optional<::mlir::AffineMap> sourceMap;
};

/// The complete channel relation of one host channel value. Every producer
/// site belongs to exactly one thread-launch body-operand binding; sequential
/// and structured mutually exclusive sites under that binding contribute to
/// the same ordered event sequence. Consumers remain a non-empty set in
/// program-discovery order.
struct ChannelRelation {
  llvm::SmallVector<ChannelEndpointBinding, 1> producers;
  llvm::SmallVector<ChannelEndpointBinding, 2> consumers;
};

/// Compute the channel relation for `hostChannel` (a channel-typed block
/// argument bound at `dataflow.thread.launch` body operands), using the exact
/// whole-program channel-topology rules. Fails on an unsupported use surface.
/// This is the single owner of the channel binding relation; consumers do not
/// re-derive it.
llvm::Expected<ChannelRelation>
computeChannelRelation(::mlir::Value hostChannel);

/// Discover every host channel of `module` exactly once and deliver its exact
/// computed relation to `callback`. This is the single shared owner of channel
/// discovery: both finalized-program validation and read-only view import route
/// through it, so neither rederives the host-channel walk or the relation.
llvm::Error forEachHostChannelRelation(
    ::mlir::ModuleOp module,
    llvm::function_ref<llvm::Error(::mlir::Value, const ChannelRelation &)>
        callback);

} // namespace dataflow

#endif // DATAFLOW_IR_DATAFLOWGRAPHVALIDATION_H
