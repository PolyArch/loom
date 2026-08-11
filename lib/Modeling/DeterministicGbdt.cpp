#include "DeterministicGbdt.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

namespace loom::evaluation::models::detail {
namespace {

constexpr std::int64_t kMagnitudeLimit = std::int64_t{1} << 40;
constexpr std::uint32_t kMaximumFeatureOrHeadCount = 1024;
constexpr std::uint64_t kMaximumTrainingRows = std::uint64_t{1} << 20;
constexpr std::uint32_t kMaximumDepth = 31;
constexpr std::uint32_t kMaximumLearningRateDenominator = 1000000000;
constexpr std::uint64_t kMaximumNodeCount = std::uint64_t{1} << 24;
constexpr std::uint32_t kMaximumTreeCount = 4096;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "deterministic_gbdt_invalid: " + message);
}

bool admittedMagnitude(std::int64_t value) {
  return value >= -kMagnitudeLimit && value <= kMagnitudeLimit;
}

llvm::Expected<std::int64_t> checkedAdd(std::int64_t lhs, std::int64_t rhs) {
  const __int128 value = static_cast<__int128>(lhs) + rhs;
  if (value < std::numeric_limits<std::int64_t>::min() ||
      value > std::numeric_limits<std::int64_t>::max())
    return invalid("integer accumulation overflowed");
  return static_cast<std::int64_t>(value);
}

llvm::Expected<std::int64_t> divideRoundTiesToEven(__int128 numerator,
                                                   std::uint64_t denominator) {
  if (denominator == 0)
    return invalid("integer division has a zero denominator");
  const bool negative = numerator < 0;
  const unsigned __int128 magnitude =
      negative ? static_cast<unsigned __int128>(-numerator)
               : static_cast<unsigned __int128>(numerator);
  unsigned __int128 quotient = magnitude / denominator;
  const unsigned __int128 remainder = magnitude % denominator;
  const unsigned __int128 twiceRemainder = remainder * 2;
  if (twiceRemainder > denominator ||
      (twiceRemainder == denominator && (quotient & 1) != 0))
    ++quotient;
  const __int128 signedValue = negative ? -static_cast<__int128>(quotient)
                                        : static_cast<__int128>(quotient);
  if (signedValue < std::numeric_limits<std::int64_t>::min() ||
      signedValue > std::numeric_limits<std::int64_t>::max())
    return invalid("rounded integer result overflowed");
  return static_cast<std::int64_t>(signedValue);
}

struct ResidualMoments final {
  std::uint64_t count = 0;
  __int128 sum = 0;
  __int128 sumOfSquares = 0;
};

void addResidual(ResidualMoments &moments, std::int64_t residual) {
  ++moments.count;
  moments.sum += residual;
  moments.sumOfSquares +=
      static_cast<__int128>(residual) * static_cast<__int128>(residual);
}

ResidualMoments moments(llvm::ArrayRef<std::uint32_t> rowOrdinals,
                        llvm::ArrayRef<std::int64_t> residuals) {
  ResidualMoments result;
  for (std::uint32_t row : rowOrdinals)
    addResidual(result, residuals[row]);
  return result;
}

ResidualMoments subtract(const ResidualMoments &total,
                         const ResidualMoments &part) {
  return {total.count - part.count, total.sum - part.sum,
          total.sumOfSquares - part.sumOfSquares};
}

llvm::Expected<std::int64_t> leafValue(const ResidualMoments &moments,
                                       const DeterministicGbdtConfig &config) {
  if (moments.count == 0)
    return invalid("leaf has no rows");
  const std::uint64_t denominator =
      moments.count * config.learningRateDenominator;
  return divideRoundTiesToEven(
      moments.sum * static_cast<__int128>(config.learningRateNumerator),
      denominator);
}

llvm::Expected<std::int64_t>
leafValue(llvm::ArrayRef<std::uint32_t> rowOrdinals,
          llvm::ArrayRef<std::int64_t> residuals,
          const DeterministicGbdtConfig &config) {
  return leafValue(moments(rowOrdinals, residuals), config);
}

__int128 squaredError(const ResidualMoments &moments, std::int64_t prediction) {
  const __int128 predicted = prediction;
  return moments.sumOfSquares - 2 * predicted * moments.sum +
         static_cast<__int128>(moments.count) * predicted * predicted;
}

struct SplitCandidate final {
  std::uint32_t featureOrdinal = 0;
  std::int64_t threshold = 0;
  __int128 error = 0;
  std::uint32_t priority = 0;
};

bool betterSplit(const SplitCandidate &candidate,
                 const SplitCandidate &selected) {
  if (candidate.error != selected.error)
    return candidate.error < selected.error;
  if (candidate.priority != selected.priority)
    return candidate.priority < selected.priority;
  if (candidate.featureOrdinal != selected.featureOrdinal)
    return candidate.featureOrdinal < selected.featureOrdinal;
  return candidate.threshold < selected.threshold;
}

class TreeBuilder final {
public:
  TreeBuilder(llvm::ArrayRef<DeterministicGbdtTrainingRow> rows,
              llvm::ArrayRef<std::int64_t> residuals,
              const DeterministicGbdtConfig &config, std::uint32_t round,
              std::uint32_t head)
      : rows_(rows), residuals_(residuals), config_(config), round_(round),
        head_(head) {}

  llvm::Expected<DeterministicGbdtTree> build() {
    std::vector<std::uint32_t> rows(rows_.size());
    std::iota(rows.begin(), rows.end(), 0);
    auto root = buildNode(rows, 0);
    if (!root)
      return root.takeError();
    (void)*root;
    return DeterministicGbdtTree{head_, std::move(nodes_)};
  }

private:
  llvm::Expected<std::uint32_t>
  buildNode(llvm::ArrayRef<std::uint32_t> rowOrdinals, std::uint32_t depth) {
    const std::uint32_t ordinal = static_cast<std::uint32_t>(nodes_.size());
    nodes_.push_back({});
    auto parentLeaf = leafValue(rowOrdinals, residuals_, config_);
    if (!parentLeaf)
      return parentLeaf.takeError();
    if (depth >= config_.maximumDepth ||
        rowOrdinals.size() <
            static_cast<std::uint64_t>(config_.minimumRowsPerLeaf) * 2) {
      nodes_[ordinal].leafValue = *parentLeaf;
      return ordinal;
    }

    const ResidualMoments total = moments(rowOrdinals, residuals_);
    const __int128 parentError = squaredError(total, *parentLeaf);
    std::optional<SplitCandidate> selected;
    for (std::uint32_t feature = 0; feature < rows_.front().features.size();
         ++feature) {
      std::vector<std::uint32_t> ordered(rowOrdinals.begin(),
                                         rowOrdinals.end());
      llvm::sort(ordered, [&](std::uint32_t lhs, std::uint32_t rhs) {
        if (rows_[lhs].features[feature] != rows_[rhs].features[feature])
          return rows_[lhs].features[feature] < rows_[rhs].features[feature];
        return lhs < rhs;
      });
      ResidualMoments left;
      for (std::size_t split = 1; split < ordered.size(); ++split) {
        addResidual(left, residuals_[ordered[split - 1]]);
        if (split < config_.minimumRowsPerLeaf ||
            ordered.size() - split < config_.minimumRowsPerLeaf)
          continue;
        const std::int64_t threshold =
            rows_[ordered[split - 1]].features[feature];
        if (threshold == rows_[ordered[split]].features[feature])
          continue;
        const ResidualMoments right = subtract(total, left);
        auto leftValue = leafValue(left, config_);
        if (!leftValue)
          return leftValue.takeError();
        auto rightValue = leafValue(right, config_);
        if (!rightValue)
          return rightValue.takeError();
        const __int128 error =
            squaredError(left, *leftValue) + squaredError(right, *rightValue);
        const std::uint64_t rotation =
            config_.seed + static_cast<std::uint64_t>(round_) * 1315423911ULL +
            static_cast<std::uint64_t>(head_) * 2654435761ULL + depth;
        SplitCandidate candidate{
            feature, threshold, error,
            static_cast<std::uint32_t>((feature + rotation) %
                                       rows_.front().features.size())};
        if (!selected || betterSplit(candidate, *selected))
          selected = candidate;
      }
    }
    if (!selected || selected->error >= parentError) {
      nodes_[ordinal].leafValue = *parentLeaf;
      return ordinal;
    }

    std::vector<std::uint32_t> leftRows;
    std::vector<std::uint32_t> rightRows;
    leftRows.reserve(rowOrdinals.size());
    rightRows.reserve(rowOrdinals.size());
    for (std::uint32_t row : rowOrdinals) {
      if (rows_[row].features[selected->featureOrdinal] <= selected->threshold)
        leftRows.push_back(row);
      else
        rightRows.push_back(row);
    }
    auto left = buildNode(leftRows, depth + 1);
    if (!left)
      return left.takeError();
    auto right = buildNode(rightRows, depth + 1);
    if (!right)
      return right.takeError();
    nodes_[ordinal] = DeterministicGbdtNode{DeterministicGbdtNode::Kind::Split,
                                            selected->featureOrdinal,
                                            selected->threshold,
                                            *left,
                                            *right,
                                            0};
    return ordinal;
  }

  llvm::ArrayRef<DeterministicGbdtTrainingRow> rows_;
  llvm::ArrayRef<std::int64_t> residuals_;
  const DeterministicGbdtConfig &config_;
  std::uint32_t round_;
  std::uint32_t head_;
  std::vector<DeterministicGbdtNode> nodes_;
};

llvm::Expected<std::int64_t>
evaluateTree(const DeterministicGbdtTree &tree,
             llvm::ArrayRef<std::int64_t> features) {
  std::uint32_t ordinal = 0;
  for (std::size_t visited = 0; visited <= tree.nodes.size(); ++visited) {
    if (ordinal >= tree.nodes.size())
      return invalid("tree traversal reached an invalid node");
    const DeterministicGbdtNode &node = tree.nodes[ordinal];
    if (node.kind == DeterministicGbdtNode::Kind::Leaf)
      return node.leafValue;
    if (node.featureOrdinal >= features.size())
      return invalid("tree split references an unavailable feature");
    ordinal = features[node.featureOrdinal] <= node.threshold ? node.leftChild
                                                              : node.rightChild;
  }
  return invalid("tree traversal contains a cycle");
}

llvm::Error validateTree(const DeterministicGbdtTree &tree,
                         std::uint32_t featureCount) {
  if (tree.nodes.empty() || tree.nodes.size() > kMaximumNodeCount)
    return invalid("tree has an invalid node count");
  const auto validateSubtree =
      [&](const auto &self, std::uint32_t ordinal,
          std::uint32_t depth) -> llvm::Expected<std::uint32_t> {
    if (ordinal >= tree.nodes.size())
      return invalid("tree child relation is out of range");
    const DeterministicGbdtNode &node = tree.nodes[ordinal];
    if (node.kind == DeterministicGbdtNode::Kind::Leaf) {
      if (node.featureOrdinal != 0 || node.threshold != 0 ||
          node.leftChild != 0 || node.rightChild != 0 ||
          !admittedMagnitude(node.leafValue))
        return invalid("leaf node is noncanonical");
      return ordinal + 1;
    }
    if (node.kind != DeterministicGbdtNode::Kind::Split ||
        node.featureOrdinal >= featureCount ||
        !admittedMagnitude(node.threshold) || node.leafValue != 0 ||
        node.leftChild != ordinal + 1 || depth >= kMaximumDepth)
      return invalid("split node is noncanonical");
    auto afterLeft = self(self, node.leftChild, depth + 1);
    if (!afterLeft)
      return afterLeft.takeError();
    if (node.rightChild != *afterLeft)
      return invalid("tree nodes are not in canonical preorder");
    return self(self, node.rightChild, depth + 1);
  };
  auto end = validateSubtree(validateSubtree, 0, 0);
  if (!end)
    return end.takeError();
  if (*end != tree.nodes.size())
    return invalid("tree contains unreachable nodes");
  return llvm::Error::success();
}

llvm::Error validateEnsemble(const DeterministicGbdtEnsemble &ensemble) {
  if (ensemble.featureCount == 0 ||
      ensemble.featureCount > kMaximumFeatureOrHeadCount ||
      ensemble.headCount == 0 ||
      ensemble.headCount > kMaximumFeatureOrHeadCount)
    return invalid("ensemble has an invalid feature or head count");
  if (ensemble.baseValues.size() != ensemble.headCount ||
      ensemble.trees.empty() ||
      (ensemble.trees.size() % ensemble.headCount) != 0)
    return invalid("ensemble does not contain complete head rounds");
  for (std::int64_t value : ensemble.baseValues)
    if (!admittedMagnitude(value))
      return invalid("ensemble base value is outside the admitted range");
  for (std::size_t index = 0; index < ensemble.trees.size(); ++index) {
    const DeterministicGbdtTree &tree = ensemble.trees[index];
    if (tree.headOrdinal != index % ensemble.headCount)
      return invalid("ensemble tree heads are not canonical");
    if (llvm::Error error = validateTree(tree, ensemble.featureCount))
      return error;
  }
  return llvm::Error::success();
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

class Decoder final {
public:
  explicit Decoder(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<std::uint32_t> u32(llvm::StringRef description) {
    if (bytes_.size() < 4)
      return invalid(description + " is truncated");
    const std::uint32_t value = (static_cast<std::uint32_t>(bytes_[0]) << 24) |
                                (static_cast<std::uint32_t>(bytes_[1]) << 16) |
                                (static_cast<std::uint32_t>(bytes_[2]) << 8) |
                                bytes_[3];
    bytes_ = bytes_.drop_front(4);
    return value;
  }

  llvm::Expected<std::uint64_t> u64(llvm::StringRef description) {
    if (bytes_.size() < 8)
      return invalid(description + " is truncated");
    std::uint64_t value = 0;
    for (std::uint8_t byte : bytes_.take_front(8))
      value = (value << 8) | byte;
    bytes_ = bytes_.drop_front(8);
    return value;
  }

  bool empty() const { return bytes_.empty(); }
  std::size_t remaining() const { return bytes_.size(); }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
};

} // namespace

llvm::Expected<DeterministicGbdtEnsemble>
trainDeterministicGbdt(llvm::ArrayRef<DeterministicGbdtTrainingRow> rows,
                       const DeterministicGbdtConfig &config,
                       const DeterministicGbdtEnsemble *initial) {
  if (rows.empty() || rows.size() > kMaximumTrainingRows)
    return invalid("training row count is outside the admitted range");
  if (config.treeCount == 0 || config.treeCount > kMaximumTreeCount ||
      config.maximumDepth == 0 || config.maximumDepth > kMaximumDepth ||
      config.minimumRowsPerLeaf == 0 ||
      config.minimumRowsPerLeaf > rows.size() ||
      config.learningRateNumerator == 0 ||
      config.learningRateDenominator == 0 ||
      config.learningRateNumerator > config.learningRateDenominator ||
      config.learningRateDenominator > kMaximumLearningRateDenominator)
    return invalid("training configuration is outside the admitted range");
  const std::size_t featureCount = rows.front().features.size();
  const std::size_t headCount = rows.front().targets.size();
  if (featureCount == 0 || featureCount > kMaximumFeatureOrHeadCount ||
      headCount == 0 || headCount > kMaximumFeatureOrHeadCount)
    return invalid("training row has an invalid feature or target count");
  for (const DeterministicGbdtTrainingRow &row : rows) {
    if (row.features.size() != featureCount || row.targets.size() != headCount)
      return invalid("training rows do not share one exact shape");
    for (std::int64_t value : row.features)
      if (!admittedMagnitude(value))
        return invalid("training feature is outside the admitted range");
    for (std::int64_t value : row.targets)
      if (!admittedMagnitude(value))
        return invalid("training target is outside the admitted range");
  }

  DeterministicGbdtEnsemble ensemble;
  std::vector<std::vector<std::int64_t>> predictions(
      headCount, std::vector<std::int64_t>(rows.size()));
  std::uint32_t roundOffset = 0;
  if (initial) {
    if (llvm::Error error = validateEnsemble(*initial))
      return std::move(error);
    if (initial->featureCount != featureCount ||
        initial->headCount != headCount)
      return invalid("initial ensemble has a different training shape");
    const std::size_t initialRounds = initial->trees.size() / headCount;
    if (initialRounds >
        std::numeric_limits<std::uint32_t>::max() - config.treeCount)
      return invalid("warm-start tree ordinal overflows uint32");
    ensemble = *initial;
    roundOffset = static_cast<std::uint32_t>(initialRounds);
    for (std::size_t row = 0; row < rows.size(); ++row) {
      auto prediction = inferDeterministicGbdt(*initial, rows[row].features);
      if (!prediction)
        return prediction.takeError();
      for (std::size_t head = 0; head < headCount; ++head)
        predictions[head][row] = (*prediction)[head];
    }
  } else {
    ensemble.featureCount = static_cast<std::uint32_t>(featureCount);
    ensemble.headCount = static_cast<std::uint32_t>(headCount);
    ensemble.baseValues.reserve(headCount);
    for (std::size_t head = 0; head < headCount; ++head) {
      __int128 sum = 0;
      for (const DeterministicGbdtTrainingRow &row : rows)
        sum += row.targets[head];
      auto base = divideRoundTiesToEven(sum, rows.size());
      if (!base)
        return base.takeError();
      ensemble.baseValues.push_back(*base);
      std::fill(predictions[head].begin(), predictions[head].end(), *base);
    }
  }

  const std::uint64_t newTreeCount =
      static_cast<std::uint64_t>(config.treeCount) * headCount;
  if (ensemble.trees.size() > kMaximumNodeCount ||
      newTreeCount > kMaximumNodeCount - ensemble.trees.size())
    return invalid("warm-start ensemble exceeds the admitted tree count");
  ensemble.trees.reserve(ensemble.trees.size() +
                         static_cast<std::size_t>(newTreeCount));
  for (std::uint32_t round = 0; round < config.treeCount; ++round) {
    for (std::uint32_t head = 0; head < headCount; ++head) {
      std::vector<std::int64_t> residuals;
      residuals.reserve(rows.size());
      for (std::size_t row = 0; row < rows.size(); ++row) {
        const __int128 residual =
            static_cast<__int128>(rows[row].targets[head]) -
            predictions[head][row];
        if (residual < -kMagnitudeLimit || residual > kMagnitudeLimit)
          return invalid("training residual is outside the admitted range");
        residuals.push_back(static_cast<std::int64_t>(residual));
      }
      TreeBuilder builder(rows, residuals, config, roundOffset + round, head);
      auto tree = builder.build();
      if (!tree)
        return tree.takeError();
      for (std::size_t row = 0; row < rows.size(); ++row) {
        auto delta = evaluateTree(*tree, rows[row].features);
        if (!delta)
          return delta.takeError();
        auto updated = checkedAdd(predictions[head][row], *delta);
        if (!updated)
          return updated.takeError();
        if (!admittedMagnitude(*updated))
          return invalid("training prediction is outside the admitted range");
        predictions[head][row] = *updated;
      }
      ensemble.trees.push_back(std::move(*tree));
    }
  }
  if (llvm::Error error = validateEnsemble(ensemble))
    return std::move(error);
  return ensemble;
}

llvm::Expected<std::vector<std::int64_t>>
inferDeterministicGbdt(const DeterministicGbdtEnsemble &ensemble,
                       llvm::ArrayRef<std::int64_t> features) {
  if (llvm::Error error = validateEnsemble(ensemble))
    return std::move(error);
  if (features.size() != ensemble.featureCount)
    return invalid("inference feature count does not match the ensemble");
  for (std::int64_t value : features)
    if (!admittedMagnitude(value))
      return invalid("inference feature is outside the admitted range");
  std::vector<std::int64_t> result = ensemble.baseValues;
  for (const DeterministicGbdtTree &tree : ensemble.trees) {
    auto delta = evaluateTree(tree, features);
    if (!delta)
      return delta.takeError();
    auto updated = checkedAdd(result[tree.headOrdinal], *delta);
    if (!updated)
      return updated.takeError();
    result[tree.headOrdinal] = *updated;
  }
  return result;
}

llvm::Expected<std::vector<std::uint8_t>>
encodeDeterministicGbdt(const DeterministicGbdtEnsemble &ensemble) {
  if (llvm::Error error = validateEnsemble(ensemble))
    return std::move(error);
  std::vector<std::uint8_t> bytes;
  appendU32(bytes, ensemble.featureCount);
  appendU32(bytes, ensemble.headCount);
  appendU64(bytes, ensemble.trees.size());
  for (std::int64_t value : ensemble.baseValues)
    appendU64(bytes, static_cast<std::uint64_t>(value));
  for (const DeterministicGbdtTree &tree : ensemble.trees) {
    appendU32(bytes, tree.headOrdinal);
    appendU64(bytes, tree.nodes.size());
    for (const DeterministicGbdtNode &node : tree.nodes) {
      appendU32(bytes, static_cast<std::uint32_t>(node.kind));
      if (node.kind == DeterministicGbdtNode::Kind::Leaf) {
        appendU64(bytes, static_cast<std::uint64_t>(node.leafValue));
      } else {
        appendU32(bytes, node.featureOrdinal);
        appendU64(bytes, static_cast<std::uint64_t>(node.threshold));
        appendU32(bytes, node.leftChild);
        appendU32(bytes, node.rightChild);
      }
    }
  }
  return bytes;
}

llvm::Expected<DeterministicGbdtEnsemble>
decodeDeterministicGbdt(llvm::ArrayRef<std::uint8_t> bytes) {
  Decoder decoder(bytes);
  auto featureCount = decoder.u32("feature count");
  if (!featureCount)
    return featureCount.takeError();
  auto headCount = decoder.u32("head count");
  if (!headCount)
    return headCount.takeError();
  auto treeCount = decoder.u64("tree count");
  if (!treeCount)
    return treeCount.takeError();
  if (*featureCount == 0 || *featureCount > kMaximumFeatureOrHeadCount ||
      *headCount == 0 || *headCount > kMaximumFeatureOrHeadCount ||
      *treeCount == 0 || *treeCount > kMaximumNodeCount)
    return invalid("ensemble header is outside the admitted range");
  DeterministicGbdtEnsemble ensemble;
  ensemble.featureCount = *featureCount;
  ensemble.headCount = *headCount;
  ensemble.baseValues.reserve(*headCount);
  for (std::uint32_t head = 0; head < *headCount; ++head) {
    auto value = decoder.u64("base value");
    if (!value)
      return value.takeError();
    ensemble.baseValues.push_back(static_cast<std::int64_t>(*value));
  }
  constexpr std::size_t minimumTreeBytes = 24;
  if (*treeCount > decoder.remaining() / minimumTreeBytes)
    return invalid("tree count exceeds the remaining payload");
  ensemble.trees.reserve(*treeCount);
  for (std::uint64_t treeOrdinal = 0; treeOrdinal < *treeCount; ++treeOrdinal) {
    auto head = decoder.u32("tree head");
    if (!head)
      return head.takeError();
    auto nodeCount = decoder.u64("tree node count");
    if (!nodeCount)
      return nodeCount.takeError();
    if (*nodeCount == 0 || *nodeCount > kMaximumNodeCount)
      return invalid("tree node count is outside the admitted range");
    constexpr std::size_t minimumNodeBytes = 12;
    if (*nodeCount > decoder.remaining() / minimumNodeBytes)
      return invalid("tree node count exceeds the remaining payload");
    DeterministicGbdtTree tree;
    tree.headOrdinal = *head;
    tree.nodes.reserve(*nodeCount);
    for (std::uint64_t nodeOrdinal = 0; nodeOrdinal < *nodeCount;
         ++nodeOrdinal) {
      auto kind = decoder.u32("node kind");
      if (!kind)
        return kind.takeError();
      if (*kind ==
          static_cast<std::uint32_t>(DeterministicGbdtNode::Kind::Leaf)) {
        auto value = decoder.u64("leaf value");
        if (!value)
          return value.takeError();
        tree.nodes.push_back({DeterministicGbdtNode::Kind::Leaf, 0, 0, 0, 0,
                              static_cast<std::int64_t>(*value)});
      } else if (*kind == static_cast<std::uint32_t>(
                              DeterministicGbdtNode::Kind::Split)) {
        auto feature = decoder.u32("split feature");
        if (!feature)
          return feature.takeError();
        auto threshold = decoder.u64("split threshold");
        if (!threshold)
          return threshold.takeError();
        auto left = decoder.u32("left child");
        if (!left)
          return left.takeError();
        auto right = decoder.u32("right child");
        if (!right)
          return right.takeError();
        tree.nodes.push_back({DeterministicGbdtNode::Kind::Split, *feature,
                              static_cast<std::int64_t>(*threshold), *left,
                              *right, 0});
      } else {
        return invalid("node has an unknown kind");
      }
    }
    ensemble.trees.push_back(std::move(tree));
  }
  if (!decoder.empty())
    return invalid("ensemble payload has trailing bytes");
  if (llvm::Error error = validateEnsemble(ensemble))
    return std::move(error);
  auto reencoded = encodeDeterministicGbdt(ensemble);
  if (!reencoded)
    return reencoded.takeError();
  if (llvm::ArrayRef(*reencoded) != bytes)
    return invalid("ensemble payload is not canonical");
  return ensemble;
}

} // namespace loom::evaluation::models::detail
