#include "Common/CanonicalRelation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

using namespace loom;

namespace {

[[noreturn]] void fail(const char *test, const std::string &message) {
  llvm::errs() << test << ": " << message << "\n";
  std::exit(EXIT_FAILURE);
}

void require(const char *test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

CanonicalRelationResult
canonicalize(const char *test, llvm::ArrayRef<std::string> vertices,
             llvm::ArrayRef<CanonicalRelationEdge> edges) {
  llvm::Expected<CanonicalRelationResult> result =
      canonicalizeRelationGraph(vertices, edges);
  if (!result)
    fail(test, llvm::toString(result.takeError()));
  return std::move(*result);
}

std::vector<std::uint8_t> bytes(const CanonicalRelationResult &result) {
  llvm::ArrayRef<std::uint8_t> value = result.bytes.bytes();
  return {value.begin(), value.end()};
}

void relabelingDoesNotChangeCanonicalResult() {
  const char *test = __func__;
  const std::vector<std::string> vertices{"root", "leaf", "leaf"};
  const std::vector<CanonicalRelationEdge> first{{0, 1, "left"},
                                                 {0, 2, "right"}};

  const std::vector<std::string> relabeled{"leaf", "root", "leaf"};
  const std::vector<CanonicalRelationEdge> second{{1, 2, "left"},
                                                  {1, 0, "right"}};

  CanonicalRelationResult a = canonicalize(test, vertices, first);
  CanonicalRelationResult b = canonicalize(test, relabeled, second);
  require(test, bytes(a) == bytes(b),
          "isomorphic vertex numbering changed canonical bytes");
}

void semanticRelationChangeChangesCanonicalResult() {
  const char *test = __func__;
  const std::vector<std::string> vertices{"source", "sink"};
  CanonicalRelationResult data = canonicalize(
      test, vertices, std::vector<CanonicalRelationEdge>{{0, 1, "data"}});
  CanonicalRelationResult control = canonicalize(
      test, vertices, std::vector<CanonicalRelationEdge>{{0, 1, "control"}});
  require(test, bytes(data) != bytes(control),
          "an edge-label semantic change did not change canonical bytes");
}

void symmetricGraphProducesACompletePermutation() {
  const char *test = __func__;
  constexpr std::uint32_t leafCount = 128;
  std::vector<std::string> vertices(leafCount + 1, "leaf");
  vertices[0] = "root";
  std::vector<CanonicalRelationEdge> edges;
  for (std::uint32_t leaf = 1; leaf <= leafCount; ++leaf)
    edges.push_back({0, leaf, "child"});

  CanonicalRelationResult result = canonicalize(test, vertices, edges);
  require(test, result.canonicalOrder.size() == vertices.size(),
          "canonical order does not cover every vertex");
  std::vector<bool> seen(vertices.size(), false);
  for (std::uint32_t vertex : result.canonicalOrder) {
    require(test, vertex < vertices.size(),
            "canonical order contains an out-of-range vertex");
    require(test, !seen[vertex], "canonical order contains a duplicate vertex");
    seen[vertex] = true;
  }
}

void materializedSymmetricOrderIsIdempotent() {
  const char *test = __func__;
  const std::vector<std::string> vertices(4, "vertex");
  const std::vector<CanonicalRelationEdge> edges{{0, 1, "next"},
                                                  {1, 2, "next"},
                                                  {2, 3, "next"},
                                                  {3, 0, "next"}};

  CanonicalRelationResult first = canonicalize(test, vertices, edges);
  std::vector<std::uint32_t> canonicalPosition(vertices.size());
  std::vector<std::string> reorderedVertices;
  reorderedVertices.reserve(vertices.size());
  for (auto [position, vertex] : llvm::enumerate(first.canonicalOrder)) {
    canonicalPosition[vertex] = position;
    reorderedVertices.push_back(vertices[vertex]);
  }
  std::vector<CanonicalRelationEdge> reorderedEdges;
  reorderedEdges.reserve(edges.size());
  for (const CanonicalRelationEdge &edge : edges)
    reorderedEdges.push_back({canonicalPosition[edge.source],
                              canonicalPosition[edge.target], edge.label});

  CanonicalRelationResult second =
      canonicalize(test, reorderedVertices, reorderedEdges);
  for (auto [position, vertex] : llvm::enumerate(second.canonicalOrder))
    require(test, position == vertex,
            "materialized symmetric canonical order was not a fixed point");
}

} // namespace

int main() {
  relabelingDoesNotChangeCanonicalResult();
  semanticRelationChangeChangesCanonicalResult();
  symmetricGraphProducesACompletePermutation();
  materializedSymmetricOrderIsIdempotent();
  llvm::outs() << "all canonical relation tests passed\n";
  return EXIT_SUCCESS;
}
