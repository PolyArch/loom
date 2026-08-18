#ifndef LOOM_TEST_APPLICATIONS_GAPBS_PAGERANK_SMOKE_H
#define LOOM_TEST_APPLICATIONS_GAPBS_PAGERANK_SMOKE_H

#include <stddef.h>
#include <stdint.h>

#define BENCHMARK_H_
#define BUILDER_H_
#define COMMAND_LINE_H_
#define GRAPH_H_
#define PVECTOR_H_

namespace std {

template <typename First, typename Second> struct pair {
  First first;
  Second second;
};

template <typename First, typename Second>
pair<First, Second> make_pair(First first, Second second) {
  return {first, second};
}

template <typename Value, size_t Capacity = 16> class vector {
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

using NodeID = int32_t;

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
  Value values_[16]{};
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

static float gapbsAbsoluteValue(float value) {
  return value < 0.0f ? -value : value;
}

__attribute__((noinline)) static int gapbsPagerankSmoke() {
  constexpr NodeID nodeCount = 4;
  static constexpr NodeID outOffsets[nodeCount + 1] = {0, 2, 3, 4, 6};
  static constexpr NodeID outNeighbors[6] = {1, 2, 2, 0, 0, 2};
  static constexpr NodeID inOffsets[nodeCount + 1] = {0, 2, 3, 6, 6};
  static constexpr NodeID inNeighbors[6] = {2, 3, 0, 0, 1, 3};
  Graph graph(nodeCount, outOffsets, outNeighbors, inOffsets, inNeighbors);
  pvector<float> scores = gapbs_pagerank_kernel(graph, 8, 0.0, false);
  constexpr float expected[nodeCount] = {0.385175735f, 0.201199681f,
                                         0.388156950f, 0.037499994f};
  float sum = 0.0f;
  for (NodeID node = 0; node < nodeCount; ++node) {
    if (gapbsAbsoluteValue(scores[node] - expected[node]) > 1.0e-6f)
      return 1;
    sum += scores[node];
  }
  return gapbsAbsoluteValue(sum - 1.01203236f) <= 1.0e-6f ? 0 : 1;
}

int main() { return gapbsPagerankSmoke(); }

#define PageRankPullGS gapbs_pagerank_kernel
#define main gapbs_upstream_main

#endif
