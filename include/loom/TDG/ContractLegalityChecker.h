#ifndef LOOM_TDG_CONTRACTLEGALITYCHECKER_H
#define LOOM_TDG_CONTRACTLEGALITYCHECKER_H

#include <cstdint>
#include <string>
#include <vector>

namespace loom {
namespace tdg {

// Describes one data-movement contract between a producer core and a consumer
// core in a multi-core task dependence graph (TDG).
struct Contract {
  unsigned producerCoreId = 0;
  unsigned consumerCoreId = 0;

  // Data volume in bytes that must flow through this contract.
  uint64_t dataBytes = 0;

  // Execution cycles of the producer kernel that generates the data.
  uint64_t producerCycles = 0;

  // Minimum number of buffer elements required at the consumer side to avoid
  // back-pressure stalls (derived from the producer's output rate and the
  // network latency).
  unsigned minBufferElements = 0;

  // Actual buffer capacity allocated by the system compiler.
  unsigned allocatedBufferElements = 0;

  // SPM (scratchpad memory) bytes requested by this contract.
  uint64_t spmBytesRequested = 0;
};

// Hardware resource budget against which contracts are checked.
struct ResourceBudget {
  // NoC bandwidth in bytes per cycle.
  double nocBandwidthBytesPerCycle = 8.0;

  // Per-core SPM budget in bytes.
  uint64_t spmBudgetBytes = 65536;

  // Maximum outstanding NoC transfers.
  unsigned maxOutstandingTransfers = 16;
};

// Identifies which legality check failed.
enum class LegalityViolation : uint8_t {
  NONE = 0,
  PRODUCER_CONSUMER_MISMATCH = 1, // producer/consumer IDs are the same
  DATA_VOLUME_ZERO = 2,           // contract moves zero bytes
  BUFFER_CAPACITY = 3,            // buffer too small for min requirement
  NOC_BANDWIDTH = 4,              // NoC bandwidth insufficient for data rate
  SPM_OVERFLOW = 5,               // SPM allocation exceeds budget
};

const char *legalityViolationName(LegalityViolation v);

// Result of a single legality check.
struct LegalityResult {
  bool legal = true;
  LegalityViolation violation = LegalityViolation::NONE;
  std::string message;
};

// Checks a set of contracts against resource budgets and structural
// constraints. Each check is self-contained and returns a LegalityResult.
class ContractLegalityChecker {
public:
  explicit ContractLegalityChecker(const ResourceBudget &budget);

  // Run all five checks on a single contract. Returns the first failure, or
  // a passing result if all checks pass.
  LegalityResult checkAll(const Contract &contract) const;

  // Run all checks on every contract in the vector. Returns the first failure
  // found across all contracts, or a passing result.
  LegalityResult checkAll(const std::vector<Contract> &contracts) const;

  // Individual checks:
  LegalityResult checkProducerConsumerMismatch(const Contract &c) const;
  LegalityResult checkDataVolumeZero(const Contract &c) const;
  LegalityResult checkBufferCapacity(const Contract &c) const;
  LegalityResult checkNocBandwidth(const Contract &c) const;
  LegalityResult checkSpmOverflow(const Contract &c) const;

private:
  ResourceBudget budget_;
};

} // namespace tdg
} // namespace loom

#endif // LOOM_TDG_CONTRACTLEGALITYCHECKER_H
