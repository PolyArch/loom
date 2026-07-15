#include "Fabric/Tech/Synthesizer/CoverageVerifier.h"

#include "Fabric/IR/ConfiguredFunction.h"
#include "Fabric/Tech/Synthesizer/Parallel.h"

#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <optional>
#include <string>

namespace loom::fabric::tech {

CoverageVerifier::CoverageVerifier(const ::loom::SynthConfig &cfg)
    : parallelMatch(cfg.coverageVerifierParallelMatch),
      parallelismWorkers(cfg.parallelismWorkers) {}

CoverageReport CoverageVerifier::verify(
    ::fabric::FuOp fu, ::llvm::ArrayRef<::fabric::ConfiguredFunction> inputs) {
  CoverageReport report;
  report.witnesses.assign(inputs.size(), std::nullopt);
  if (!fu)
    return report;

  ::llvm::SmallVector<::fabric::ConfiguredFunction, 8> candidates;
  std::string projectionError;
  if (::mlir::failed(::fabric::projectConfiguredFunctions(fu, candidates,
                                                          projectionError)))
    return report;

  auto matchOne = [&](size_t inputIndex) {
    for (size_t encodingIndex = 0; encodingIndex < candidates.size();
         ++encodingIndex) {
      ::fabric::ConfiguredFunctionMatch match;
      if (!::fabric::matchConfiguredFunctions(
              inputs[inputIndex], candidates[encodingIndex],
              /*preserveFuBoundaryIdentity=*/false, &match))
        continue;
      CoverageWitness witness;
      witness.encodingIndex = encodingIndex;
      for (unsigned node : match.nodeMap)
        witness.actorToFabricOp.push_back(
            candidates[encodingIndex].nodes[node].fabricResource);
      witness.inputPorts = std::move(match.inputPorts);
      witness.outputPorts = std::move(match.outputPorts);
      report.witnesses[inputIndex] = std::move(witness);
      return;
    }
  };

  if (parallelMatch && inputs.size() > 1) {
    WorkerPool pool(parallelismWorkers);
    ::llvm::SmallVector<size_t, 8> indices;
    for (size_t index = 0; index < inputs.size(); ++index)
      indices.push_back(index);
    pool.parallelFor(indices, matchOne);
  } else {
    for (size_t index = 0; index < inputs.size(); ++index)
      matchOne(index);
  }
  return report;
}

} // namespace loom::fabric::tech
