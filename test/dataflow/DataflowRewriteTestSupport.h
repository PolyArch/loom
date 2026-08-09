#ifndef LOOM_TEST_DATAFLOW_DATAFLOW_REWRITE_TEST_SUPPORT_H
#define LOOM_TEST_DATAFLOW_DATAFLOW_REWRITE_TEST_SUPPORT_H

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Simulator/DFGSimulator.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <map>
#include <string>

namespace dataflow::test {

struct FunctionalOutcome final {
  std::string status;
  llvm::SmallVector<std::string> values;
  llvm::SmallVector<llvm::SmallVector<std::string>> streams;
  std::map<std::string, llvm::SmallVector<std::string>> memoryState;
  std::map<std::string, std::string> memoryRoots;

  friend bool operator==(const FunctionalOutcome &lhs,
                         const FunctionalOutcome &rhs) {
    return lhs.status == rhs.status && lhs.values == rhs.values &&
           lhs.streams == rhs.streams && lhs.memoryState == rhs.memoryState &&
           lhs.memoryRoots == rhs.memoryRoots;
  }
};

inline llvm::Expected<FunctionalOutcome>
simulateGraph(const CanonicalDataflowArtifact &artifact, GraphRef graphRef,
              llvm::ArrayRef<loom::sim::DFGRuntimeArg> args = {}) {
  auto view = artifact.view();
  if (!view)
    return view.takeError();
  auto graph = view->resolve(graphRef);
  if (!graph)
    return graph.takeError();
  loom::sim::DFGSimulationOptions options;
  options.graphName =
      mlir::cast<dataflow::GraphOp>(graph->op).getSymName().str();
  options.args.append(args.begin(), args.end());
  auto report = loom::sim::simulateDataflowGraph(artifact.module(), options);
  if (!report)
    return report.takeError();
  return FunctionalOutcome{report->status, report->finalOutputs,
                           report->finalStreamOutputs, report->finalMemoryState,
                           report->finalMemoryRoots};
}

inline llvm::Expected<FunctionalOutcome>
simulateOnlyGraph(const CanonicalDataflowArtifact &artifact,
                  llvm::ArrayRef<loom::sim::DFGRuntimeArg> args = {}) {
  auto view = artifact.view();
  if (!view)
    return view.takeError();
  if (view->graphs().size() != 1)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "rewrite differential fixture does not contain exactly one graph");
  return simulateGraph(artifact, view->graphs().front().ref, args);
}

} // namespace dataflow::test

#endif // LOOM_TEST_DATAFLOW_DATAFLOW_REWRITE_TEST_SUPPORT_H
