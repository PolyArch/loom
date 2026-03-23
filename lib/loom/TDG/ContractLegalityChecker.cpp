#include "loom/TDG/ContractLegalityChecker.h"

#include <sstream>

namespace loom {
namespace tdg {

const char *legalityViolationName(LegalityViolation v) {
  switch (v) {
  case LegalityViolation::NONE:
    return "NONE";
  case LegalityViolation::PRODUCER_CONSUMER_MISMATCH:
    return "PRODUCER_CONSUMER_MISMATCH";
  case LegalityViolation::DATA_VOLUME_ZERO:
    return "DATA_VOLUME_ZERO";
  case LegalityViolation::BUFFER_CAPACITY:
    return "BUFFER_CAPACITY";
  case LegalityViolation::NOC_BANDWIDTH:
    return "NOC_BANDWIDTH";
  case LegalityViolation::SPM_OVERFLOW:
    return "SPM_OVERFLOW";
  }
  return "UNKNOWN";
}

ContractLegalityChecker::ContractLegalityChecker(const ResourceBudget &budget)
    : budget_(budget) {}

LegalityResult ContractLegalityChecker::checkAll(const Contract &contract) const {
  LegalityResult r;

  r = checkProducerConsumerMismatch(contract);
  if (!r.legal)
    return r;

  r = checkDataVolumeZero(contract);
  if (!r.legal)
    return r;

  r = checkBufferCapacity(contract);
  if (!r.legal)
    return r;

  r = checkNocBandwidth(contract);
  if (!r.legal)
    return r;

  r = checkSpmOverflow(contract);
  if (!r.legal)
    return r;

  return r;
}

LegalityResult
ContractLegalityChecker::checkAll(const std::vector<Contract> &contracts) const {
  for (const auto &c : contracts) {
    LegalityResult r = checkAll(c);
    if (!r.legal)
      return r;
  }
  return LegalityResult{};
}

// Check 1: producer and consumer core IDs must differ.
LegalityResult
ContractLegalityChecker::checkProducerConsumerMismatch(const Contract &c) const {
  LegalityResult r;
  if (c.producerCoreId == c.consumerCoreId) {
    r.legal = false;
    r.violation = LegalityViolation::PRODUCER_CONSUMER_MISMATCH;
    std::ostringstream oss;
    oss << "producer and consumer are the same core (id=" << c.producerCoreId
        << "); intra-core transfers do not use NoC contracts";
    r.message = oss.str();
  }
  return r;
}

// Check 2: data volume must be non-zero for a meaningful contract.
LegalityResult
ContractLegalityChecker::checkDataVolumeZero(const Contract &c) const {
  LegalityResult r;
  if (c.dataBytes == 0) {
    r.legal = false;
    r.violation = LegalityViolation::DATA_VOLUME_ZERO;
    r.message = "contract data volume is zero bytes";
  }
  return r;
}

// Check 3: the allocated buffer capacity must meet or exceed the minimum
// number of buffer elements required to avoid back-pressure stalls.
LegalityResult
ContractLegalityChecker::checkBufferCapacity(const Contract &c) const {
  LegalityResult r;
  if (c.allocatedBufferElements < c.minBufferElements) {
    r.legal = false;
    r.violation = LegalityViolation::BUFFER_CAPACITY;
    std::ostringstream oss;
    oss << "allocated buffer capacity (" << c.allocatedBufferElements
        << " elements) is less than the minimum required ("
        << c.minBufferElements << " elements)";
    r.message = oss.str();
  }
  return r;
}

// Check 4: the NoC must have enough bandwidth to sustain the data rate
// required by the contract. The required rate is dataBytes / producerCycles
// (bytes per cycle). If this exceeds the budget's nocBandwidthBytesPerCycle,
// the contract is illegal.
LegalityResult
ContractLegalityChecker::checkNocBandwidth(const Contract &c) const {
  LegalityResult r;

  // If producerCycles is zero, the rate is effectively infinite, which always
  // exceeds the budget (unless dataBytes is also zero, caught by check 2).
  if (c.producerCycles == 0) {
    r.legal = false;
    r.violation = LegalityViolation::NOC_BANDWIDTH;
    r.message = "producer cycles is zero; cannot sustain infinite data rate";
    return r;
  }

  double requiredRate =
      static_cast<double>(c.dataBytes) / static_cast<double>(c.producerCycles);

  if (requiredRate > budget_.nocBandwidthBytesPerCycle) {
    r.legal = false;
    r.violation = LegalityViolation::NOC_BANDWIDTH;
    std::ostringstream oss;
    oss << "contract requires " << requiredRate
        << " bytes/cycle but NoC budget is "
        << budget_.nocBandwidthBytesPerCycle << " bytes/cycle";
    r.message = oss.str();
  }
  return r;
}

// Check 5: the SPM bytes requested by this contract must not exceed the
// per-core SPM budget.
LegalityResult
ContractLegalityChecker::checkSpmOverflow(const Contract &c) const {
  LegalityResult r;
  if (c.spmBytesRequested > budget_.spmBudgetBytes) {
    r.legal = false;
    r.violation = LegalityViolation::SPM_OVERFLOW;
    std::ostringstream oss;
    oss << "contract requests " << c.spmBytesRequested
        << " SPM bytes but per-core budget is " << budget_.spmBudgetBytes
        << " bytes";
    r.message = oss.str();
  }
  return r;
}

} // namespace tdg
} // namespace loom
