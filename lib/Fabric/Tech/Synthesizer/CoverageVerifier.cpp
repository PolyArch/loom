#include "Fabric/Tech/Synthesizer/CoverageVerifier.h"

#include "Fabric/IR/ConfiguredFunction.h"
#include "Fabric/Tech/Synthesizer/Parallel.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
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

  ::llvm::SmallVector<::fabric::ConfiguredFunctionKey, 8> candidateKeys;
  candidateKeys.reserve(candidates.size());
  ::llvm::DenseMap<std::uint64_t, ::llvm::SmallVector<size_t, 2>>
      candidatesByHash;
  for (auto [index, candidate] : ::llvm::enumerate(candidates)) {
    candidateKeys.push_back(::fabric::getConfiguredFunctionKey(
        candidate, /*preserveFuBoundaryIdentity=*/false));
    candidatesByHash[candidateKeys.back().hash].push_back(index);
  }
  ::llvm::SmallVector<::fabric::ConfiguredFunctionKey, 8> inputKeys;
  inputKeys.reserve(inputs.size());
  for (const ::fabric::ConfiguredFunction &input : inputs)
    inputKeys.push_back(::fabric::getConfiguredFunctionKey(
        input, /*preserveFuBoundaryIdentity=*/false));

  auto matchOne = [&](size_t inputIndex) {
    const ::fabric::ConfiguredFunctionKey &inputKey = inputKeys[inputIndex];
    auto matches = candidatesByHash.find(inputKey.hash);
    if (matches == candidatesByHash.end())
      return;
    for (size_t encodingIndex : matches->second) {
      if (candidateKeys[encodingIndex].canonical != inputKey.canonical)
        continue;
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
