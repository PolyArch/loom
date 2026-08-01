#ifndef LOOM_LIB_DATAFLOW_IR_DATAFLOWGRAPHCAUSALITY_H
#define LOOM_LIB_DATAFLOW_IR_DATAFLOWGRAPHCAUSALITY_H

#include "mlir/IR/Value.h"

#include <memory>

namespace dataflow::detail {

class GraphCausalDependencyCache {
public:
  GraphCausalDependencyCache();
  ~GraphCausalDependencyCache();

  GraphCausalDependencyCache(const GraphCausalDependencyCache &) = delete;
  GraphCausalDependencyCache &
  operator=(const GraphCausalDependencyCache &) = delete;

  bool dependsOn(mlir::Value event, mlir::Value prerequisite);

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace dataflow::detail

#endif
