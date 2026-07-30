#include "Common/CanonicalRelation.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace loom {
namespace {

void appendU64(std::string &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<char>(value >> shift));
}

void appendBytes(std::string &bytes, llvm::StringRef value) {
  appendU64(bytes, value.size());
  bytes.append(value.data(), value.size());
}

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "canonical_relation_invalid: " + message);
}

class Canonicalizer {
public:
  static llvm::Expected<Canonicalizer>
  build(llvm::ArrayRef<std::string> vertexIntrinsics,
        llvm::ArrayRef<CanonicalRelationEdge> edges) {
    if (vertexIntrinsics.size() > std::numeric_limits<std::uint32_t>::max())
      return invalid("vertex count exceeds the persistent ordinal range");
    if (edges.size() > std::numeric_limits<std::uint32_t>::max())
      return invalid("edge count exceeds the internal ordinal range");

    Canonicalizer result(vertexIntrinsics, edges);
    for (const CanonicalRelationEdge &edge : edges) {
      if (edge.source >= vertexIntrinsics.size() ||
          edge.target >= vertexIntrinsics.size())
        return invalid("edge endpoint is outside the vertex inventory");
    }
    result.buildAdjacency();
    result.buildEdgeLabelRanks();
    return result;
  }

  CanonicalRelationResult canonicalize() const {
    std::map<std::string, std::uint64_t> ranks;
    for (const std::string &intrinsic : intrinsics_)
      ranks.emplace(intrinsic, 0);
    std::uint64_t next = 0;
    for (auto &entry : ranks)
      entry.second = next++;

    std::vector<std::uint64_t> initial(intrinsics_.size());
    for (std::uint32_t vertex = 0; vertex < intrinsics_.size(); ++vertex)
      initial[vertex] = ranks[intrinsics_[vertex]];

    std::vector<std::uint32_t> path;
    Leaf leaf = search(std::move(initial), path);
    return {CanonicalSemanticBytes(std::move(leaf.bytes)),
            std::move(leaf.order)};
  }

private:
  struct Leaf {
    std::vector<std::uint8_t> bytes;
    std::vector<std::uint32_t> order;
  };

  Canonicalizer(llvm::ArrayRef<std::string> vertexIntrinsics,
                llvm::ArrayRef<CanonicalRelationEdge> edges)
      : intrinsics_(vertexIntrinsics.begin(), vertexIntrinsics.end()),
        edges_(edges.begin(), edges.end()) {}

  void buildAdjacency() {
    outgoing_.assign(intrinsics_.size(), {});
    incoming_.assign(intrinsics_.size(), {});
    for (std::uint32_t ordinal = 0; ordinal < edges_.size(); ++ordinal) {
      outgoing_[edges_[ordinal].source].push_back(ordinal);
      incoming_[edges_[ordinal].target].push_back(ordinal);
    }
  }

  void buildEdgeLabelRanks() {
    std::vector<std::uint32_t> order(edges_.size());
    std::iota(order.begin(), order.end(), 0);
    llvm::sort(order, [&](std::uint32_t lhs, std::uint32_t rhs) {
      return compareEncodedLabel(edges_[lhs].label, edges_[rhs].label) < 0;
    });

    edgeLabelRanks_.resize(edges_.size());
    std::uint32_t rank = 0;
    if (!order.empty())
      edgeLabelRanks_[order.front()] = rank;
    for (std::size_t index = 1; index < order.size(); ++index) {
      if (compareEncodedLabel(edges_[order[index - 1]].label,
                              edges_[order[index]].label) != 0)
        ++rank;
      edgeLabelRanks_[order[index]] = rank;
    }
  }

  struct RefinementNeighbor {
    std::uint8_t direction;
    std::uint32_t labelRank;
    std::uint64_t color;
  };

  struct RefinementSignature {
    std::uint64_t color;
    llvm::SmallVector<RefinementNeighbor, 4> neighbors;
  };

  static int compareEncodedLabel(llvm::StringRef lhs, llvm::StringRef rhs) {
    if (lhs.size() != rhs.size())
      return lhs.size() < rhs.size() ? -1 : 1;
    return lhs.compare(rhs);
  }

  static int compareNeighbor(const RefinementNeighbor &lhs,
                             const RefinementNeighbor &rhs) {
    if (lhs.direction != rhs.direction)
      return lhs.direction < rhs.direction ? -1 : 1;
    if (lhs.labelRank != rhs.labelRank)
      return lhs.labelRank < rhs.labelRank ? -1 : 1;
    if (lhs.color != rhs.color)
      return lhs.color < rhs.color ? -1 : 1;
    return 0;
  }

  static int compareSignature(const RefinementSignature &lhs,
                              const RefinementSignature &rhs) {
    if (lhs.color != rhs.color)
      return lhs.color < rhs.color ? -1 : 1;
    if (lhs.neighbors.size() != rhs.neighbors.size())
      return lhs.neighbors.size() < rhs.neighbors.size() ? -1 : 1;
    for (auto [left, right] : llvm::zip(lhs.neighbors, rhs.neighbors))
      if (int order = compareNeighbor(left, right))
        return order;
    return 0;
  }

  std::vector<std::uint64_t> refine(std::vector<std::uint64_t> colors) const {
    while (true) {
      std::vector<RefinementSignature> signatures;
      signatures.reserve(intrinsics_.size());
      for (std::uint32_t vertex = 0; vertex < intrinsics_.size(); ++vertex) {
        RefinementSignature signature{colors[vertex], {}};
        signature.neighbors.reserve(outgoing_[vertex].size() +
                                    incoming_[vertex].size());
        for (std::uint32_t ordinal : outgoing_[vertex]) {
          const CanonicalRelationEdge &edge = edges_[ordinal];
          signature.neighbors.push_back(
              {0, edgeLabelRanks_[ordinal], colors[edge.target]});
        }
        for (std::uint32_t ordinal : incoming_[vertex]) {
          const CanonicalRelationEdge &edge = edges_[ordinal];
          signature.neighbors.push_back(
              {1, edgeLabelRanks_[ordinal], colors[edge.source]});
        }
        llvm::sort(signature.neighbors, [](const RefinementNeighbor &lhs,
                                           const RefinementNeighbor &rhs) {
          return compareNeighbor(lhs, rhs) < 0;
        });
        signatures.push_back(std::move(signature));
      }

      const std::uint64_t previousClassCount =
          colors.empty()
              ? 0
              : *llvm::max_element(colors) + static_cast<std::uint64_t>(1);
      std::vector<llvm::SmallVector<std::uint32_t>> cells(
          static_cast<std::size_t>(previousClassCount));
      for (std::uint32_t vertex = 0; vertex < intrinsics_.size(); ++vertex)
        cells[colors[vertex]].push_back(vertex);

      std::vector<std::uint64_t> refined(colors.size());
      std::uint64_t rank = 0;
      bool hasRank = false;
      for (auto &cell : cells) {
        if (cell.empty())
          continue;
        llvm::sort(cell, [&](std::uint32_t lhs, std::uint32_t rhs) {
          return compareSignature(signatures[lhs], signatures[rhs]) < 0;
        });
        if (hasRank)
          ++rank;
        hasRank = true;
        refined[cell.front()] = rank;
        for (std::size_t index = 1; index < cell.size(); ++index) {
          if (compareSignature(signatures[cell[index - 1]],
                               signatures[cell[index]]) != 0)
            ++rank;
          refined[cell[index]] = rank;
        }
      }
      const std::uint64_t next = hasRank ? rank + 1 : 0;
      colors = std::move(refined);
      if (next == previousClassCount)
        return colors;
    }
  }

  std::vector<std::uint8_t>
  serialize(llvm::ArrayRef<std::uint32_t> order) const {
    std::vector<std::uint64_t> rank(intrinsics_.size());
    for (std::uint64_t position = 0; position < order.size(); ++position)
      rank[order[position]] = position;

    std::string bytes;
    appendU64(bytes, order.size());
    for (std::uint32_t vertex : order) {
      appendBytes(bytes, intrinsics_[vertex]);
      std::vector<std::pair<std::string, std::uint64_t>> outgoing;
      outgoing.reserve(outgoing_[vertex].size());
      for (std::uint32_t ordinal : outgoing_[vertex]) {
        const CanonicalRelationEdge &edge = edges_[ordinal];
        outgoing.emplace_back(edge.label, rank[edge.target]);
      }
      llvm::sort(outgoing);
      appendU64(bytes, outgoing.size());
      for (const auto &relation : outgoing) {
        appendBytes(bytes, relation.first);
        appendU64(bytes, relation.second);
      }
    }
    return {bytes.begin(), bytes.end()};
  }

  bool isTranspositionAutomorphism(std::uint32_t lhs, std::uint32_t rhs) const {
    if (intrinsics_[lhs] != intrinsics_[rhs])
      return false;

    llvm::SmallVector<std::uint32_t, 16> incident;
    incident.append(outgoing_[lhs].begin(), outgoing_[lhs].end());
    incident.append(incoming_[lhs].begin(), incoming_[lhs].end());
    incident.append(outgoing_[rhs].begin(), outgoing_[rhs].end());
    incident.append(incoming_[rhs].begin(), incoming_[rhs].end());
    llvm::sort(incident);
    incident.erase(std::unique(incident.begin(), incident.end()),
                   incident.end());

    struct Relation {
      std::uint32_t source;
      std::uint32_t target;
      llvm::StringRef label;
    };
    auto less = [](const Relation &a, const Relation &b) {
      if (a.source != b.source)
        return a.source < b.source;
      if (a.target != b.target)
        return a.target < b.target;
      return a.label.compare(b.label) < 0;
    };
    auto equal = [](const Relation &a, const Relation &b) {
      return a.source == b.source && a.target == b.target && a.label == b.label;
    };
    auto transpose = [&](std::uint32_t vertex) {
      if (vertex == lhs)
        return rhs;
      if (vertex == rhs)
        return lhs;
      return vertex;
    };

    llvm::SmallVector<Relation, 16> original;
    llvm::SmallVector<Relation, 16> transposed;
    original.reserve(incident.size());
    transposed.reserve(incident.size());
    for (std::uint32_t ordinal : incident) {
      const CanonicalRelationEdge &edge = edges_[ordinal];
      original.push_back({edge.source, edge.target, edge.label});
      transposed.push_back(
          {transpose(edge.source), transpose(edge.target), edge.label});
    }
    llvm::sort(original, less);
    llvm::sort(transposed, less);
    return std::equal(original.begin(), original.end(), transposed.begin(),
                      equal);
  }

  Leaf search(std::vector<std::uint64_t> colors,
              std::vector<std::uint32_t> &path) const {
    colors = refine(std::move(colors));

    const std::size_t cellCount =
        colors.empty()
            ? 0
            : static_cast<std::size_t>(*llvm::max_element(colors) + 1);
    std::vector<llvm::SmallVector<std::uint32_t>> cells(cellCount);
    for (std::uint32_t vertex = 0; vertex < intrinsics_.size(); ++vertex)
      cells[colors[vertex]].push_back(vertex);

    const llvm::SmallVector<std::uint32_t> *target = nullptr;
    for (const auto &cell : cells)
      if (cell.size() > 1) {
        target = &cell;
        break;
      }

    if (!target) {
      Leaf leaf;
      leaf.order.resize(intrinsics_.size());
      for (std::uint32_t vertex = 0; vertex < intrinsics_.size(); ++vertex)
        leaf.order[vertex] = vertex;
      llvm::sort(leaf.order, [&](std::uint32_t lhs, std::uint32_t rhs) {
        return colors[lhs] < colors[rhs];
      });
      leaf.bytes = serialize(leaf.order);
      return leaf;
    }

    std::vector<std::uint32_t> parent(intrinsics_.size());
    for (std::uint32_t vertex = 0; vertex < intrinsics_.size(); ++vertex)
      parent[vertex] = vertex;
    auto find = [&](std::uint32_t value) {
      while (parent[value] != value) {
        parent[value] = parent[parent[value]];
        value = parent[value];
      }
      return value;
    };
    auto unite = [&](std::uint32_t lhs, std::uint32_t rhs) {
      parent[find(lhs)] = find(rhs);
    };
    auto applyAutomorphism =
        [&](const std::vector<std::uint32_t> &permutation) {
          for (std::uint32_t ancestor : path)
            if (permutation[ancestor] != ancestor)
              return;
          for (std::uint32_t vertex = 0; vertex < intrinsics_.size(); ++vertex)
            unite(vertex, permutation[vertex]);
        };
    for (std::size_t lhs = 0; lhs < target->size(); ++lhs) {
      const std::uint32_t representative = (*target)[lhs];
      if (find(representative) != representative)
        continue;
      for (std::size_t rhs = lhs + 1; rhs < target->size(); ++rhs) {
        const std::uint32_t candidate = (*target)[rhs];
        if (find(candidate) == find(representative))
          continue;
        if (isTranspositionAutomorphism(representative, candidate))
          unite(candidate, representative);
      }
    }
    for (const std::vector<std::uint32_t> &permutation : automorphisms_)
      applyAutomorphism(permutation);

    Leaf best;
    bool hasBest = false;
    std::vector<bool> exploredOrbits(intrinsics_.size(), false);
    for (std::uint32_t candidate : *target) {
      const std::uint32_t orbit = find(candidate);
      if (exploredOrbits[orbit])
        continue;
      exploredOrbits[orbit] = true;
      std::vector<std::uint64_t> individualized = colors;
      const std::uint64_t targetColor = individualized[candidate];
      for (std::uint64_t &color : individualized)
        color = color * 2 + 1;
      individualized[candidate] = targetColor * 2;
      path.push_back(candidate);
      Leaf leaf = search(std::move(individualized), path);
      path.pop_back();
      if (!hasBest || leaf.bytes < best.bytes) {
        best = std::move(leaf);
        hasBest = true;
      } else if (leaf.bytes == best.bytes) {
        std::vector<std::uint32_t> permutation(intrinsics_.size());
        for (std::uint32_t position = 0; position < intrinsics_.size();
             ++position)
          permutation[leaf.order[position]] = best.order[position];
        applyAutomorphism(permutation);
        automorphisms_.push_back(std::move(permutation));
      }
    }
    return best;
  }

  std::vector<std::string> intrinsics_;
  std::vector<CanonicalRelationEdge> edges_;
  std::vector<std::uint32_t> edgeLabelRanks_;
  std::vector<llvm::SmallVector<std::uint32_t>> outgoing_;
  std::vector<llvm::SmallVector<std::uint32_t>> incoming_;
  mutable std::vector<std::vector<std::uint32_t>> automorphisms_;
};

} // namespace

llvm::Expected<CanonicalRelationResult>
canonicalizeRelationGraph(llvm::ArrayRef<std::string> vertexIntrinsics,
                          llvm::ArrayRef<CanonicalRelationEdge> edges) {
  llvm::Expected<Canonicalizer> canonicalizer =
      Canonicalizer::build(vertexIntrinsics, edges);
  if (!canonicalizer)
    return canonicalizer.takeError();
  return canonicalizer->canonicalize();
}

} // namespace loom
