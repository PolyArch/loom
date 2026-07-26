#include "Common/CanonicalRelation.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace loom {
namespace {

void appendU8(std::string &bytes, std::uint8_t value) {
  bytes.push_back(static_cast<char>(value));
}

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

    Canonicalizer result(vertexIntrinsics, edges);
    for (const CanonicalRelationEdge &edge : edges) {
      if (edge.source >= vertexIntrinsics.size() ||
          edge.target >= vertexIntrinsics.size())
        return invalid("edge endpoint is outside the vertex inventory");
    }
    result.buildAdjacency();
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

  std::vector<std::uint64_t> refine(std::vector<std::uint64_t> colors) const {
    while (true) {
      std::vector<std::string> signatures(intrinsics_.size());
      for (std::uint32_t vertex = 0; vertex < intrinsics_.size(); ++vertex) {
        std::string signature;
        appendU64(signature, colors[vertex]);
        llvm::SmallVector<std::string> neighbors;
        neighbors.reserve(outgoing_[vertex].size() + incoming_[vertex].size());
        for (std::uint32_t ordinal : outgoing_[vertex]) {
          const CanonicalRelationEdge &edge = edges_[ordinal];
          std::string neighbor;
          appendU8(neighbor, 0);
          appendBytes(neighbor, edge.label);
          appendU64(neighbor, colors[edge.target]);
          neighbors.push_back(std::move(neighbor));
        }
        for (std::uint32_t ordinal : incoming_[vertex]) {
          const CanonicalRelationEdge &edge = edges_[ordinal];
          std::string neighbor;
          appendU8(neighbor, 1);
          appendBytes(neighbor, edge.label);
          appendU64(neighbor, colors[edge.source]);
          neighbors.push_back(std::move(neighbor));
        }
        llvm::sort(neighbors);
        appendU64(signature, neighbors.size());
        for (const std::string &neighbor : neighbors)
          signature.append(neighbor);
        signatures[vertex] = std::move(signature);
      }

      std::map<std::string, std::uint64_t> ranks;
      for (const std::string &signature : signatures)
        ranks.emplace(signature, 0);
      std::uint64_t next = 0;
      for (auto &entry : ranks)
        entry.second = next++;

      const std::size_t previousClassCount =
          std::set<std::uint64_t>(colors.begin(), colors.end()).size();
      for (std::uint32_t vertex = 0; vertex < intrinsics_.size(); ++vertex)
        colors[vertex] = ranks[signatures[vertex]];
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

  Leaf search(std::vector<std::uint64_t> colors,
              std::vector<std::uint32_t> &path) const {
    colors = refine(std::move(colors));

    std::map<std::uint64_t, llvm::SmallVector<std::uint32_t>> cells;
    for (std::uint32_t vertex = 0; vertex < intrinsics_.size(); ++vertex)
      cells[colors[vertex]].push_back(vertex);

    const llvm::SmallVector<std::uint32_t> *target = nullptr;
    for (const auto &cell : cells)
      if (cell.second.size() > 1) {
        target = &cell.second;
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
    std::function<std::uint32_t(std::uint32_t)> find =
        [&](std::uint32_t value) {
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
    for (const std::vector<std::uint32_t> &permutation : automorphisms_)
      applyAutomorphism(permutation);

    Leaf best;
    bool hasBest = false;
    std::set<std::uint32_t> exploredOrbits;
    for (std::uint32_t candidate : *target) {
      if (!exploredOrbits.insert(find(candidate)).second)
        continue;
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
