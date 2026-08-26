#ifndef LOOM_FRONTEND_LOWERING_GRAPH_STREAM_BOUNDARY_LOWERING_H
#define LOOM_FRONTEND_LOWERING_GRAPH_STREAM_BOUNDARY_LOWERING_H

#include "Dataflow/IR/DataflowOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace loom {
namespace lowering {
namespace detail {

struct StreamBoundaryInfo {
  ::llvm::SmallVector<::mlir::BlockArgument, 4> inputChannels;
  ::llvm::SmallVector<::mlir::Value, 4> inputPayloads;
  ::llvm::SmallVector<::mlir::BlockArgument, 4> outputChannels;

  bool isTransient() const {
    return !inputChannels.empty() || !outputChannels.empty();
  }
};

struct StreamScheduleNode {
  enum class Kind { Empty, Endpoint, Sequence, Choice, Repeat };

  StreamScheduleNode(Kind kind, ::mlir::Location loc) : kind(kind), loc(loc) {}

  Kind kind;
  unsigned siteCount = 0;
  ::mlir::Operation *endpoint = nullptr;
  ::mlir::Operation *choice = nullptr;
  ::mlir::Operation *repeat = nullptr;
  ::mlir::Location loc;
  std::vector<std::unique_ptr<StreamScheduleNode>> children;
};

struct StreamBindingPlan {
  ::mlir::Block *scope = nullptr;
  unsigned recurrenceBits = 0;
  std::unique_ptr<StreamScheduleNode> schedule;
};

struct StreamScheduleMaterialization {
  struct ChoiceSelectorUse {
    ::mlir::Operation *choice;
    ::mlir::Operation *user;
  };

  struct RepeatSelectorUse {
    ::mlir::Operation *repeat;
    ::mlir::Operation *user;
    ::mlir::Operation *placeholder;
  };

  ::llvm::SmallVector<::mlir::Operation *, 4> endpoints;
  ::llvm::SmallVector<ChoiceSelectorUse, 4> choiceSelectorUses;
  ::llvm::SmallVector<RepeatSelectorUse, 4> repeatSelectorUses;
  ::mlir::Value activation;
  ::mlir::Value phase;
  ::mlir::Value ordinal;
  ::mlir::Value event;
  ::mlir::Value activity;
  ::mlir::Value inactiveEvent;
  ::mlir::Value close;
};

struct StreamSelectiveRouter {
  struct Node {
    ::mlir::Value selector;
    unsigned leftCount;
  };

  unsigned leafCount = 0;
  ::llvm::SmallVector<Node, 4> nodes;
};

struct StreamSelectiveRouterMaterialization {
  StreamSelectiveRouter router;
  ::llvm::SmallVector<::mlir::Value, 4> events;
};

::llvm::SmallVector<::mlir::Operation *, 4>
collectStreamEndpoints(::mlir::Value channel, bool input);

::mlir::FailureOr<StreamBoundaryInfo>
analyzeStreamBoundary(::dataflow::GraphOp graph);

::mlir::FailureOr<std::unique_ptr<StreamBindingPlan>>
analyzeStreamBinding(::mlir::Value channel, bool input, unsigned indexBits);

std::optional<std::string>
explainStreamScheduleRejection(::mlir::Operation *scope, unsigned indexBits);

::mlir::LogicalResult
checkStreamBoundaryUses(::dataflow::GraphOp graph,
                        const StreamBoundaryInfo &boundary, unsigned indexBits);

StreamScheduleMaterialization
materializeStreamSchedule(const StreamScheduleNode &schedule,
                          unsigned recurrenceBits, ::mlir::Value execution,
                          ::mlir::OpBuilder &builder,
                          ::mlir::Operation *anchor);

StreamSelectiveRouterMaterialization
materializeStreamSelectiveRouter(::mlir::Value ordinal, ::mlir::Value event,
                                 unsigned endpointCount, ::mlir::Location loc,
                                 ::mlir::OpBuilder &builder,
                                 ::mlir::Operation *anchor);

::llvm::SmallVector<::mlir::Value, 4>
routeStreamInput(const StreamSelectiveRouter &router, ::mlir::Value event,
                 ::mlir::Value input, ::mlir::Location loc,
                 ::mlir::OpBuilder &builder, ::mlir::Operation *anchor);

::mlir::Value collectStreamOutput(const StreamSelectiveRouter &router,
                                  ::mlir::ValueRange inputs,
                                  ::mlir::Location loc,
                                  ::mlir::OpBuilder &builder,
                                  ::mlir::Operation *anchor);

} // namespace detail
} // namespace lowering
} // namespace loom

#endif // LOOM_FRONTEND_LOWERING_GRAPH_STREAM_BOUNDARY_LOWERING_H
