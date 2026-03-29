#ifndef LOOM_SYSTEMCOMPILER_CONTRACT_H
#define LOOM_SYSTEMCOMPILER_CONTRACT_H

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace llvm {
namespace json {
class Value;
} // namespace json
} // namespace llvm

namespace loom {

//===----------------------------------------------------------------------===//
// Enums for TDC contract dimensions
//===----------------------------------------------------------------------===//

/// Ordering dimension: specifies element delivery order on an edge.
enum class Ordering {
  FIFO,      ///< Elements arrive in producer-emit order
  UNORDERED, ///< Elements may arrive in any order
  SYMBOLIC   ///< Ordering specified by a symbolic expression
};

/// Placement dimension: specifies memory placement for an edge buffer.
/// Replaces the former "Visibility" enum (adds AUTO, renames EXTERNAL_DRAM).
enum class Placement {
  LOCAL_SPM, ///< Local scratchpad memory
  SHARED_L2, ///< Shared L2 cache/memory
  EXTERNAL,  ///< External DRAM
  AUTO       ///< Compiler chooses placement automatically
};

//===----------------------------------------------------------------------===//
// Legacy enum aliases (downstream compatibility during migration)
//===----------------------------------------------------------------------===//

/// Legacy alias: Visibility is now Placement.
/// The old EXTERNAL_DRAM value maps to Placement::EXTERNAL.
using Visibility = Placement;

/// Legacy Backpressure enum -- retained as a migration shim.
/// Not part of TDC edge dimensions; will be removed in a future cleanup.
enum class Backpressure { BLOCK, DROP, OVERWRITE };

/// Legacy Writeback enum -- retained as a migration shim.
enum class Writeback { EAGER, LAZY };

/// Legacy Prefetch enum -- retained as a migration shim.
enum class Prefetch { NONE, NEXT_TILE, DOUBLE_BUFFER };

//===----------------------------------------------------------------------===//
// TDCEdgeSpec: edge-scope contract (Ordering, Throughput, Placement, Shape)
//===----------------------------------------------------------------------===//

/// Edge contract specification for inter-kernel communication.
/// Carries the 4 atomic TDC edge dimensions plus 3 identity fields.
struct TDCEdgeSpec {
  // --- Required identity fields ---
  std::string producerKernel;
  std::string consumerKernel;
  std::string dataTypeName; ///< Serializable type name (e.g. "f32", "i64")

  // --- Optional edge contract dimensions ---
  std::optional<Ordering> ordering;      ///< Element delivery order
  std::optional<std::string> throughput; ///< Symbolic throughput expression
  std::optional<Placement> placement;    ///< Memory placement
  std::optional<std::string> shape;      ///< Symbolic shape (e.g. "[128, N]")
};

//===----------------------------------------------------------------------===//
// TDCPathSpec: path-scope contract (Latency)
//===----------------------------------------------------------------------===//

/// Path contract specification for multi-edge latency constraints.
/// Carries the Latency dimension across a chain of edges.
struct TDCPathSpec {
  // --- Path endpoint identification (start edge and end edge) ---
  std::string startProducer;  ///< Producer kernel of the start edge
  std::string startConsumer;  ///< Consumer kernel of the start edge
  std::string endProducer;    ///< Producer kernel of the end edge
  std::string endConsumer;    ///< Consumer kernel of the end edge

  // --- Path dimension ---
  std::string latency; ///< Symbolic latency expression (upper bound cycles)
};

//===----------------------------------------------------------------------===//
// Legacy ContractSpec (retained for downstream compilation)
//===----------------------------------------------------------------------===//

/// Legacy 21-field contract specification.
/// Retained so that downstream components (L1CoreAssignment, BendersHelpers,
/// BufferAllocator, NoCScheduler, ExecutionModel, etc.) continue to compile
/// during the migration to TDCEdgeSpec. New code should use TDCEdgeSpec.
struct ContractSpec {
  // --- Structural ---
  std::string producerKernel;
  std::string consumerKernel;
  std::string dataTypeName; // Serializable type name (e.g. "f32", "i64")

  // --- Ordering ---
  Ordering ordering = Ordering::FIFO;

  // --- Rate and granularity ---
  std::optional<int64_t> productionRate;  // elements per invocation
  std::optional<int64_t> consumptionRate; // elements per invocation
  std::optional<std::pair<int64_t, int64_t>> steadyStateRatio; // num/den
  std::vector<int64_t> tileShape;

  // --- Buffering ---
  int64_t minBufferElements = 0;
  int64_t maxBufferElements = 0;
  Backpressure backpressure = Backpressure::BLOCK;
  bool doubleBuffering = false;

  // --- Memory visibility (legacy: uses Placement via Visibility alias) ---
  Visibility visibility = Visibility::LOCAL_SPM;
  Writeback producerWriteback = Writeback::EAGER;
  Prefetch consumerPrefetch = Prefetch::NONE;

  // --- Transformation permissions ---
  bool mayFuse = true;
  bool mayReplicate = true;
  bool mayPipeline = true;
  bool mayReorder = false;
  bool mayRetile = true;

  // --- Populated by L2 compiler (achieved values) ---
  std::optional<int64_t> achievedProductionRate;
  std::optional<int64_t> achievedConsumptionRate;
  std::optional<int64_t> achievedBufferSize;
};

//===----------------------------------------------------------------------===//
// Enum <-> string conversion helpers
//===----------------------------------------------------------------------===//

const char *orderingToString(Ordering o);
Ordering orderingFromString(const std::string &s);

const char *placementToString(Placement p);
Placement placementFromString(const std::string &s);

/// Legacy visibility converters delegate to placement converters.
const char *visibilityToString(Visibility v);
Visibility visibilityFromString(const std::string &s);

/// Legacy converters -- retained for downstream migration.
const char *backpressureToString(Backpressure b);
Backpressure backpressureFromString(const std::string &s);
const char *writebackToString(Writeback w);
Writeback writebackFromString(const std::string &s);
const char *prefetchToString(Prefetch p);
Prefetch prefetchFromString(const std::string &s);

//===----------------------------------------------------------------------===//
// JSON serialization -- new TDC types
//===----------------------------------------------------------------------===//

llvm::json::Value tdcEdgeSpecToJSON(const TDCEdgeSpec &spec);
TDCEdgeSpec tdcEdgeSpecFromJSON(const llvm::json::Value &v);

llvm::json::Value tdcPathSpecToJSON(const TDCPathSpec &spec);
TDCPathSpec tdcPathSpecFromJSON(const llvm::json::Value &v);

//===----------------------------------------------------------------------===//
// JSON serialization -- legacy ContractSpec
//===----------------------------------------------------------------------===//

llvm::json::Value contractSpecToJSON(const ContractSpec &spec);
ContractSpec contractSpecFromJSON(const llvm::json::Value &v);

//===----------------------------------------------------------------------===//
// Utility: shape expression parser
//===----------------------------------------------------------------------===//

/// Parse a shape expression string (e.g. "[128, hidden_dim / num_heads, 64]")
/// into a vector of individual dimension expression strings.
/// Returns an empty vector for "[]" or empty input.
std::vector<std::string> parseShapeExpr(const std::string &shapeStr);

} // namespace loom

#endif
