#ifndef TAPESTRY_TASK_GRAPH_H
#define TAPESTRY_TASK_GRAPH_H

#include <cassert>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tapestry {

// ============================================================================
// Enums
// ============================================================================

/// Ordering semantics on an inter-kernel data edge.
enum class Ordering { FIFO, UNORDERED };

/// Memory-hierarchy visibility for edge data.
enum class Visibility { LOCAL_SPM, SHARED_L2, EXTERNAL_DRAM };

/// Back-pressure policy when a producer outruns a consumer.
enum class Backpressure { BLOCK, DROP, OVERWRITE };

/// Execution target for a kernel node.
enum class ExecutionTarget { CGRA, HOST, AUTO_DETECT };

// Enum <-> string helpers (implemented in task_graph.cpp).
const char *orderingToString(Ordering o);
Ordering orderingFromString(const std::string &s);
const char *visibilityToString(Visibility v);
Visibility visibilityFromString(const std::string &s);
const char *backpressureToString(Backpressure bp);
Backpressure backpressureFromString(const std::string &s);
const char *executionTargetToString(ExecutionTarget t);
ExecutionTarget executionTargetFromString(const std::string &s);

// ============================================================================
// Contract -- edge-attached contract data (all fields optional)
// ============================================================================

struct Contract {
  std::optional<Ordering> ordering;
  std::optional<std::string> dataTypeName;
  std::optional<int64_t> rate;
  std::optional<std::vector<int64_t>> tileShape;
  std::optional<Visibility> visibility;
  std::optional<bool> doubleBuffering;
  std::optional<Backpressure> backpressure;

  // Transformation permissions (defaults match the plan).
  bool mayFuse = true;
  bool mayReplicate = true;
  bool mayPipeline = true;
  bool mayReorder = false;
  bool mayRetile = true;
};

// ============================================================================
// KernelProvenance -- metadata for locating a kernel in source/IR
// ============================================================================

struct KernelProvenance {
  std::string functionName; // Name used for LLVM-IR lookup
  std::string sourcePath;   // Path to .c/.cpp source
  void *funcPtr = nullptr;  // Optional live function pointer (Tier 1)
};

// ============================================================================
// EdgeKey -- identifies an edge by producer/consumer index pair
// ============================================================================

using EdgeKey = std::pair<unsigned, unsigned>;

// ============================================================================
// Forward declarations
// ============================================================================

class TaskGraph;

// ============================================================================
// EdgeHandle -- lightweight, copyable handle to one edge in the graph
// ============================================================================

class EdgeHandle {
public:
  EdgeHandle() = default;

  // Chainable contract setters -- each returns *this.
  EdgeHandle &ordering(Ordering o);
  EdgeHandle &data_type(const std::string &typeName);
  EdgeHandle &rate(int64_t r);
  EdgeHandle &tile_shape(std::vector<int64_t> shape);
  EdgeHandle &visibility(Visibility v);
  EdgeHandle &double_buffering(bool enable);
  EdgeHandle &backpressure(Backpressure bp);
  EdgeHandle &may_fuse(bool b);
  EdgeHandle &may_replicate(bool b);
  EdgeHandle &may_pipeline(bool b);
  EdgeHandle &may_reorder(bool b);
  EdgeHandle &may_retile(bool b);

  /// Template helper: infer data-type name from C++ type.
  template <typename T> EdgeHandle &data_type() {
    if constexpr (std::is_same_v<T, float>)
      return data_type("f32");
    else if constexpr (std::is_same_v<T, double>)
      return data_type("f64");
    else if constexpr (std::is_same_v<T, int32_t>)
      return data_type("i32");
    else if constexpr (std::is_same_v<T, int64_t>)
      return data_type("i64");
    else if constexpr (std::is_same_v<T, int16_t>)
      return data_type("i16");
    else if constexpr (std::is_same_v<T, int8_t>)
      return data_type("i8");
    else if constexpr (std::is_same_v<T, uint8_t>)
      return data_type("u8");
    else if constexpr (std::is_same_v<T, uint16_t>)
      return data_type("u16");
    else if constexpr (std::is_same_v<T, uint32_t>)
      return data_type("u32");
    else if constexpr (std::is_same_v<T, uint64_t>)
      return data_type("u64");
    else
      static_assert(!std::is_same_v<T, T>,
                    "Unsupported data type for EdgeHandle::data_type<T>()");
    return *this; // unreachable but satisfies the compiler
  }

  /// Read-only access to the underlying contract.
  const Contract &contract() const;

  /// Producer and consumer kernel names.
  const std::string &producerName() const;
  const std::string &consumerName() const;

private:
  friend class TaskGraph;
  EdgeHandle(TaskGraph *graph, EdgeKey key) : graph_(graph), key_(key) {}

  TaskGraph *graph_ = nullptr;
  EdgeKey key_;
};

// ============================================================================
// KernelHandle -- lightweight, copyable handle to one kernel node
// ============================================================================

class KernelHandle {
public:
  KernelHandle() = default;

  /// Set the execution target for this kernel.
  KernelHandle &target(ExecutionTarget t);

  /// Get the kernel name.
  const std::string &name() const;

  /// Get the execution target.
  ExecutionTarget executionTarget() const;

  /// Get the kernel provenance.
  const KernelProvenance &provenance() const;

  /// Internal index (used by TaskGraph internals).
  unsigned index() const { return index_; }

private:
  friend class TaskGraph;
  KernelHandle(TaskGraph *graph, unsigned idx)
      : graph_(graph), index_(idx) {}

  TaskGraph *graph_ = nullptr;
  unsigned index_ = 0;
};

// ============================================================================
// Internal map types
// ============================================================================

/// Stores per-kernel metadata.
struct KernelInfo {
  std::string name;
  KernelProvenance provenance;
  ExecutionTarget target = ExecutionTarget::AUTO_DETECT;
};

/// EdgeMap: contract attributes keyed by (producer_idx, consumer_idx).
using EdgeMap = std::map<EdgeKey, Contract>;

/// KernelMap: kernel info keyed by index.
using KernelMap = std::vector<KernelInfo>;

// ============================================================================
// TaskGraph -- top-level graph container
//
// Conceptually wraps tf::Taskflow for graph storage.  Internally uses a
// lightweight C++17 adjacency-list representation so the public header does
// not require Taskflow C++20 headers.  The adjacency list is semantically
// identical to calling tf::Taskflow::emplace() + Task::precede().
// ============================================================================

class TaskGraph {
public:
  explicit TaskGraph(const std::string &name);
  ~TaskGraph();

  // Non-copyable, movable.
  TaskGraph(const TaskGraph &) = delete;
  TaskGraph &operator=(const TaskGraph &) = delete;
  TaskGraph(TaskGraph &&) noexcept;
  TaskGraph &operator=(TaskGraph &&) noexcept;

  // -----------------------------------------------------------------------
  // Kernel definition
  // -----------------------------------------------------------------------

  /// Add a kernel backed by a function pointer.
  /// The template accepts any callable/function-pointer type.
  template <typename F>
  KernelHandle kernel(const std::string &kernelName, F funcPtr) {
    KernelInfo info;
    info.name = kernelName;
    info.provenance.functionName = kernelName;
    info.provenance.funcPtr = reinterpret_cast<void *>(funcPtr);
    return addKernelImpl(std::move(info));
  }

  /// Add a kernel by name only (no function pointer -- used by auto_analyze).
  KernelHandle kernel(const std::string &kernelName);

  // -----------------------------------------------------------------------
  // Edge definition
  // -----------------------------------------------------------------------

  /// Connect two kernels with a data-flow edge.  Returns a chainable handle.
  EdgeHandle connect(KernelHandle producer, KernelHandle consumer);

  /// Look up an existing edge by kernel names.
  EdgeHandle edge(const std::string &producerName,
                  const std::string &consumerName);

  // -----------------------------------------------------------------------
  // Inspection
  // -----------------------------------------------------------------------

  /// Print topology and contract summary to stdout.
  void dump() const;

  /// Dump the graph as a DOT-format string.
  std::string dumpDot() const;

  /// Number of kernels in the graph.
  size_t numKernels() const;

  /// Number of edges in the graph.
  size_t numEdges() const;

  /// Iterate over all kernels.  Visitor signature: void(const KernelInfo&).
  void forEachKernel(
      std::function<void(const KernelInfo &)> visitor) const;

  /// Iterate over all edges.
  /// Visitor signature: void(const std::string& producer,
  ///                         const std::string& consumer,
  ///                         const Contract&).
  void forEachEdge(
      std::function<void(const std::string &, const std::string &,
                         const Contract &)> visitor) const;

  // -----------------------------------------------------------------------
  // Internal access (for kernel_compiler, tdg_emitter, compile)
  // -----------------------------------------------------------------------

  /// Edge-contract map.
  const EdgeMap &edges() const;

  /// Kernel metadata.
  const KernelMap &kernels() const;

  /// Graph name.
  const std::string &name() const;

  /// Resolve a kernel name to its index.  Returns (unsigned)-1 on failure.
  unsigned kernelIndex(const std::string &kernelName) const;

  /// Get the adjacency list (successors of each kernel by index).
  const std::vector<std::vector<unsigned>> &adjacency() const;

  /// Get predecessor list (predecessors of each kernel by index).
  const std::vector<std::vector<unsigned>> &predecessors() const;

private:
  KernelHandle addKernelImpl(KernelInfo info);

  // Access contract by key (used by EdgeHandle).
  friend class EdgeHandle;
  friend class KernelHandle;
  Contract &contractRef(EdgeKey key);
  const Contract &contractRef(EdgeKey key) const;
  const KernelInfo &kernelRef(unsigned idx) const;
  KernelInfo &kernelRef(unsigned idx);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace tapestry

#endif // TAPESTRY_TASK_GRAPH_H
