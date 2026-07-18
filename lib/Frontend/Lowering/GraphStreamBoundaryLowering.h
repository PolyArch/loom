#ifndef LOOM_FRONTEND_LOWERING_GRAPH_STREAM_BOUNDARY_LOWERING_H
#define LOOM_FRONTEND_LOWERING_GRAPH_STREAM_BOUNDARY_LOWERING_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <vector>

namespace loom {
namespace lowering {
namespace detail {

struct StreamScheduleNode {
  enum class Kind { Endpoint, Sequence, Choice };

  StreamScheduleNode(Kind kind, ::mlir::Location loc) : kind(kind), loc(loc) {}

  Kind kind;
  unsigned width = 0;
  ::mlir::Operation *endpoint = nullptr;
  ::mlir::Operation *choice = nullptr;
  ::mlir::Value selector;
  ::mlir::Location loc;
  std::vector<std::unique_ptr<StreamScheduleNode>> children;
};

struct StreamBindingPlan {
  ::mlir::Block *scope = nullptr;
  ::mlir::Value channel;
  bool input = false;
  unsigned boundaryIndex = 0;
  std::unique_ptr<StreamScheduleNode> schedule;
};

::llvm::SmallVector<::mlir::Operation *, 4>
collectStreamEndpoints(::mlir::Value channel, bool input);

::mlir::FailureOr<std::unique_ptr<StreamBindingPlan>>
analyzeStreamBinding(::mlir::Value channel, bool input);

} // namespace detail
} // namespace lowering
} // namespace loom

#endif // LOOM_FRONTEND_LOWERING_GRAPH_STREAM_BOUNDARY_LOWERING_H
