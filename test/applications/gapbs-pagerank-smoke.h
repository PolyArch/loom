#ifndef LOOM_TEST_APPLICATIONS_GAPBS_PAGERANK_SMOKE_H
#define LOOM_TEST_APPLICATIONS_GAPBS_PAGERANK_SMOKE_H

#include "../../include/Runtime/Computation.h"

#include <stddef.h>
#include <stdint.h>

#if defined(LOOM_APPLICATION_HOST_EXECUTION)
#include <stdio.h>
#endif

#if !defined(LOOM_PAGERANK_ITERATIONS)
#define LOOM_PAGERANK_ITERATIONS 8
#endif

/* Graph extent. The smoke input keeps the smallest graph that still exercises
   an irregular pull-direction gather. A qualified input scales the same
   generated graph until the measured computation dominates the System's fixed
   per-launch cost, so the saturation target it declares is reachable at all. */
#if !defined(LOOM_PAGERANK_NODE_COUNT)
#define LOOM_PAGERANK_NODE_COUNT 4
#endif

#define BENCHMARK_H_
#define BUILDER_H_
#define COMMAND_LINE_H_
#define GRAPH_H_
#define PVECTOR_H_

using NodeID = int32_t;

/* One deterministic generated graph shared by every input. Out-degrees cycle
   through one, two, and three edges so the pull-direction gather stays
   irregular and some nodes keep an empty in-neighborhood. Both strides are odd
   and therefore coprime with the power-of-two node counts this portfolio
   selects, so one node's out-neighbours stay distinct. */
enum : NodeID {
  PAGERANK_NODE_COUNT = LOOM_PAGERANK_NODE_COUNT,
  PAGERANK_MAXIMUM_DEGREE = 3,
  PAGERANK_NEIGHBOR_STRIDE = 13,
  PAGERANK_SOURCE_STRIDE = 7,
  PAGERANK_EDGE_CAPACITY = PAGERANK_NODE_COUNT * PAGERANK_MAXIMUM_DEGREE,
};

static_assert(PAGERANK_NODE_COUNT >= PAGERANK_MAXIMUM_DEGREE,
              "every generated out-neighborhood must fit in the node domain");

namespace std {

template <typename First, typename Second> struct pair {
  First first;
  Second second;
};

template <typename First, typename Second>
pair<First, Second> make_pair(First first, Second second) {
  return {first, second};
}

template <typename Value, size_t Capacity = PAGERANK_NODE_COUNT> class vector {
public:
  vector(size_t size = 0) : size_(size) {}
  Value &operator[](size_t index) { return values_[index]; }
  const Value &operator[](size_t index) const { return values_[index]; }
  Value *begin() { return values_; }
  Value *end() { return values_ + size_; }
  const Value *begin() const { return values_; }
  const Value *end() const { return values_ + size_; }

private:
  Value values_[Capacity]{};
  size_t size_ = 0;
};

struct EndLine final {};
struct OutputStream final {
  template <typename Value> OutputStream &operator<<(const Value &) {
    return *this;
  }
};
static OutputStream cout;
static EndLine endl;

} // namespace std

template <typename Value> class pvector {
public:
  explicit pvector(size_t size) : size_(size) {}
  pvector(size_t size, Value initial) : size_(size) { fill(initial); }
  pvector(const pvector &) = delete;
  pvector(pvector &&other) : size_(other.size_) {
    for (size_t index = 0; index < size_; ++index)
      values_[index] = other.values_[index];
  }
  Value &operator[](size_t index) { return values_[index]; }
  const Value &operator[](size_t index) const { return values_[index]; }
  Value *begin() { return values_; }
  Value *end() { return values_ + size_; }
  const Value *begin() const { return values_; }
  const Value *end() const { return values_ + size_; }
  size_t size() const { return size_; }
  void fill(Value value) {
    for (size_t index = 0; index < size_; ++index)
      values_[index] = value;
  }

private:
  Value values_[PAGERANK_NODE_COUNT]{};
  size_t size_ = 0;
};

class Graph final {
public:
  class Neighborhood final {
  public:
    Neighborhood(const NodeID *begin, const NodeID *end)
        : begin_(begin), end_(end) {}
    const NodeID *begin() const { return begin_; }
    const NodeID *end() const { return end_; }

  private:
    const NodeID *begin_;
    const NodeID *end_;
  };

  class Vertices final {
  public:
    class Iterator final {
    public:
      explicit Iterator(NodeID value) : value_(value) {}
      NodeID operator*() const { return value_; }
      Iterator &operator++() {
        ++value_;
        return *this;
      }
      bool operator!=(const Iterator &other) const {
        return value_ != other.value_;
      }

    private:
      NodeID value_;
    };

    explicit Vertices(NodeID count) : count_(count) {}
    Iterator begin() const { return Iterator(0); }
    Iterator end() const { return Iterator(count_); }

  private:
    NodeID count_;
  };

  Graph() = default;
  Graph(NodeID nodeCount, const NodeID *outOffsets, const NodeID *outNeighbors,
        const NodeID *inOffsets, const NodeID *inNeighbors)
      : nodeCount_(nodeCount), outOffsets_(outOffsets),
        outNeighbors_(outNeighbors), inOffsets_(inOffsets),
        inNeighbors_(inNeighbors) {}
  NodeID num_nodes() const { return nodeCount_; }
  NodeID out_degree(NodeID node) const {
    return outOffsets_[node + 1] - outOffsets_[node];
  }
  Neighborhood out_neigh(NodeID node) const {
    return {outNeighbors_ + outOffsets_[node],
            outNeighbors_ + outOffsets_[node + 1]};
  }
  Neighborhood in_neigh(NodeID node) const {
    return {inNeighbors_ + inOffsets_[node],
            inNeighbors_ + inOffsets_[node + 1]};
  }
  Vertices vertices() const { return Vertices(nodeCount_); }

private:
  NodeID nodeCount_ = 0;
  const NodeID *outOffsets_ = nullptr;
  const NodeID *outNeighbors_ = nullptr;
  const NodeID *inOffsets_ = nullptr;
  const NodeID *inNeighbors_ = nullptr;
};

inline float fabs(float value) { return value < 0.0f ? -value : value; }
inline double fabs(double value) { return value < 0.0 ? -value : value; }
inline void PrintStep(int, double) {}
inline void PrintTime(const char *, double) {}

template <typename Key, typename Value>
std::vector<std::pair<Value, Key>>
TopK(const std::vector<std::pair<Key, Value>> &, size_t) {
  return {};
}

class CLPageRank final {
public:
  CLPageRank(int, char **, const char *, double, int) {}
  bool ParseArgs() const { return false; }
  int max_iters() const { return 0; }
  double tolerance() const { return 0.0; }
  bool logging_en() const { return false; }
};

class Builder final {
public:
  explicit Builder(const CLPageRank &) {}
  Graph MakeGraph() const { return {}; }
};

template <typename... Arguments> void BenchmarkKernel(Arguments &&...) {}

extern "C" pvector<float> gapbs_pagerank_kernel(const Graph &graph,
                                                int maximumIterations,
                                                double epsilon,
                                                bool loggingEnabled);

static NodeID gapbsOutOffsets[PAGERANK_NODE_COUNT + 1];
static NodeID gapbsOutNeighbors[PAGERANK_EDGE_CAPACITY];
static NodeID gapbsInOffsets[PAGERANK_NODE_COUNT + 1];
static NodeID gapbsInNeighbors[PAGERANK_EDGE_CAPACITY];
static NodeID gapbsInCursor[PAGERANK_NODE_COUNT];
static float gapbsReferenceScores[PAGERANK_NODE_COUNT];
static float gapbsReferenceContributions[PAGERANK_NODE_COUNT];

static float gapbsAbsoluteValue(float value) {
  return value < 0.0f ? -value : value;
}

/* The upstream kernel's damping factor, mirrored so the reference below and
   `PageRankPullGS` cannot drift apart. */
static const float gapbsDampingFactor = 0.85f;

static NodeID gapbsOutDegree(NodeID node) {
  return 1 + node % PAGERANK_MAXIMUM_DEGREE;
}

/* Both compressed neighbourhoods are derived from one out-edge rule, so the
   pull-direction kernel and the push-direction generator never disagree. */
static void gapbsBuildGraph() {
  NodeID edgeCount = 0;
  for (NodeID node = 0; node < PAGERANK_NODE_COUNT; ++node) {
    gapbsOutOffsets[node] = edgeCount;
    const NodeID degree = gapbsOutDegree(node);
    for (NodeID index = 0; index < degree; ++index)
      gapbsOutNeighbors[edgeCount++] =
          (node * PAGERANK_SOURCE_STRIDE + index * PAGERANK_NEIGHBOR_STRIDE +
           1) %
          PAGERANK_NODE_COUNT;
  }
  gapbsOutOffsets[PAGERANK_NODE_COUNT] = edgeCount;
  for (NodeID node = 0; node <= PAGERANK_NODE_COUNT; ++node)
    gapbsInOffsets[node] = 0;
  for (NodeID edge = 0; edge < edgeCount; ++edge)
    ++gapbsInOffsets[gapbsOutNeighbors[edge] + 1];
  for (NodeID node = 0; node < PAGERANK_NODE_COUNT; ++node)
    gapbsInOffsets[node + 1] += gapbsInOffsets[node];
  for (NodeID node = 0; node < PAGERANK_NODE_COUNT; ++node)
    gapbsInCursor[node] = gapbsInOffsets[node];
  for (NodeID node = 0; node < PAGERANK_NODE_COUNT; ++node)
    for (NodeID edge = gapbsOutOffsets[node]; edge < gapbsOutOffsets[node + 1];
         ++edge)
      gapbsInNeighbors[gapbsInCursor[gapbsOutNeighbors[edge]]++] = node;
}

/* An independent host implementation of the same Gauss-Seidel pull iteration,
   evaluated outside the measured computation interval. It replaces the score
   constants this harness used to transcribe for one fixed graph, so every
   extent keeps an exact per-node oracle. */
__attribute__((noinline)) static void gapbsPagerankReference(const Graph &graph,
                                                             int iterations) {
  const NodeID nodeCount = graph.num_nodes();
  const float initialScore = 1.0f / nodeCount;
  const float baseScore = (1.0f - gapbsDampingFactor) / nodeCount;
  for (NodeID node = 0; node < nodeCount; ++node) {
    gapbsReferenceScores[node] = initialScore;
    gapbsReferenceContributions[node] = initialScore / graph.out_degree(node);
  }
  for (int iteration = 0; iteration < iterations; ++iteration)
    for (NodeID node = 0; node < nodeCount; ++node) {
      float incomingTotal = 0.0f;
      for (NodeID source : graph.in_neigh(node))
        incomingTotal += gapbsReferenceContributions[source];
      gapbsReferenceScores[node] =
          baseScore + gapbsDampingFactor * incomingTotal;
      gapbsReferenceContributions[node] =
          gapbsReferenceScores[node] / graph.out_degree(node);
    }
}

__attribute__((noinline)) static int gapbsPagerankApplication() {
  gapbsBuildGraph();
  Graph graph(PAGERANK_NODE_COUNT, gapbsOutOffsets, gapbsOutNeighbors,
              gapbsInOffsets, gapbsInNeighbors);
  loom_computation_begin();
  pvector<float> scores =
      gapbs_pagerank_kernel(graph, LOOM_PAGERANK_ITERATIONS, 0.0, false);
  loom_computation_end();
  gapbsPagerankReference(graph, LOOM_PAGERANK_ITERATIONS);
  float sum = 0.0f;
  for (NodeID node = 0; node < PAGERANK_NODE_COUNT; ++node) {
    const float expected = gapbsReferenceScores[node];
    /* Every score is at least the positive base score, so one relative bound
       covers every extent. It admits accelerator reduction reordering and
       still rejects a wrong gather. */
    if (gapbsAbsoluteValue(scores[node] - expected) > 1.0e-5f * expected)
      return 1;
    sum += scores[node];
  }
#if defined(LOOM_APPLICATION_HOST_EXECUTION)
  printf("pagerank nodes: %d\n", (int)PAGERANK_NODE_COUNT);
  printf("pagerank iterations: %d\n", LOOM_PAGERANK_ITERATIONS);
  printf("pagerank sum: %.6f\n", (double)sum);
#endif
  return 0;
}

int main() { return gapbsPagerankApplication(); }

#define PageRankPullGS gapbs_pagerank_kernel
#define main gapbs_upstream_main

#endif
