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

enum class Ordering { FIFO, UNORDERED, AFFINE_INDEXED };

enum class Backpressure { BLOCK, DROP, OVERWRITE };

enum class Visibility { LOCAL_SPM, SHARED_L2, EXTERNAL_DRAM };

enum class Writeback { EAGER, LAZY };

enum class Prefetch { NONE, NEXT_TILE, DOUBLE_BUFFER };

/// Core contract specification for inter-kernel communication.
/// Contains both specified fields (from the user/TDG) and achieved fields
/// (populated by the L2 compiler after mapping).
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

  // --- Memory visibility ---
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

// Enum <-> string conversion helpers
const char *orderingToString(Ordering o);
Ordering orderingFromString(const std::string &s);
const char *backpressureToString(Backpressure b);
Backpressure backpressureFromString(const std::string &s);
const char *visibilityToString(Visibility v);
Visibility visibilityFromString(const std::string &s);
const char *writebackToString(Writeback w);
Writeback writebackFromString(const std::string &s);
const char *prefetchToString(Prefetch p);
Prefetch prefetchFromString(const std::string &s);

// JSON serialization
llvm::json::Value contractSpecToJSON(const ContractSpec &spec);
ContractSpec contractSpecFromJSON(const llvm::json::Value &v);

} // namespace loom

#endif
