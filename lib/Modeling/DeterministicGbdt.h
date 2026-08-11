#ifndef LOOM_LIB_MODELING_DETERMINISTICGBDT_H
#define LOOM_LIB_MODELING_DETERMINISTICGBDT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::evaluation::models::detail {

struct DeterministicGbdtConfig final {
  std::uint64_t seed = 0;
  std::uint32_t treeCount = 0;
  std::uint32_t maximumDepth = 0;
  std::uint32_t minimumRowsPerLeaf = 0;
  std::uint32_t learningRateNumerator = 0;
  std::uint32_t learningRateDenominator = 0;
};

struct DeterministicGbdtTrainingRow final {
  std::vector<std::int64_t> features;
  std::vector<std::int64_t> targets;
};

struct DeterministicGbdtNode final {
  enum class Kind : std::uint32_t { Leaf = 0, Split = 1 };

  Kind kind = Kind::Leaf;
  std::uint32_t featureOrdinal = 0;
  std::int64_t threshold = 0;
  std::uint32_t leftChild = 0;
  std::uint32_t rightChild = 0;
  std::int64_t leafValue = 0;

  friend bool operator==(const DeterministicGbdtNode &lhs,
                         const DeterministicGbdtNode &rhs) {
    return lhs.kind == rhs.kind && lhs.featureOrdinal == rhs.featureOrdinal &&
           lhs.threshold == rhs.threshold && lhs.leftChild == rhs.leftChild &&
           lhs.rightChild == rhs.rightChild && lhs.leafValue == rhs.leafValue;
  }
};

struct DeterministicGbdtTree final {
  std::uint32_t headOrdinal = 0;
  std::vector<DeterministicGbdtNode> nodes;

  friend bool operator==(const DeterministicGbdtTree &lhs,
                         const DeterministicGbdtTree &rhs) {
    return lhs.headOrdinal == rhs.headOrdinal && lhs.nodes == rhs.nodes;
  }
};

struct DeterministicGbdtEnsemble final {
  std::uint32_t featureCount = 0;
  std::uint32_t headCount = 0;
  std::vector<std::int64_t> baseValues;
  std::vector<DeterministicGbdtTree> trees;

  friend bool operator==(const DeterministicGbdtEnsemble &lhs,
                         const DeterministicGbdtEnsemble &rhs) {
    return lhs.featureCount == rhs.featureCount &&
           lhs.headCount == rhs.headCount && lhs.baseValues == rhs.baseValues &&
           lhs.trees == rhs.trees;
  }
};

llvm::Expected<DeterministicGbdtEnsemble>
trainDeterministicGbdt(llvm::ArrayRef<DeterministicGbdtTrainingRow> rows,
                       const DeterministicGbdtConfig &config,
                       const DeterministicGbdtEnsemble *initial = nullptr);

llvm::Expected<std::vector<std::int64_t>>
inferDeterministicGbdt(const DeterministicGbdtEnsemble &ensemble,
                       llvm::ArrayRef<std::int64_t> features);

llvm::Expected<std::vector<std::uint8_t>>
encodeDeterministicGbdt(const DeterministicGbdtEnsemble &ensemble);

llvm::Expected<DeterministicGbdtEnsemble>
decodeDeterministicGbdt(llvm::ArrayRef<std::uint8_t> bytes);

} // namespace loom::evaluation::models::detail

#endif // LOOM_LIB_MODELING_DETERMINISTICGBDT_H
