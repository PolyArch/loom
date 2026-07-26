#ifndef LOOM_FABRIC_IR_FUCAPABILITYDOMAIN_H
#define LOOM_FABRIC_IR_FUCAPABILITYDOMAIN_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace fabric {

/// One static selector choice in an FU capability template. The node ordinal
/// addresses a fabric.mux or fabric.demux in the owning FU's physical-node
/// order. selectedPort is an input ordinal for mux and an output ordinal for
/// demux.
struct FuCapabilityRouteSelection final {
  std::uint64_t selectorNodeOrdinal = 0;
  std::uint64_t selectedPort = 0;

  friend bool operator==(const FuCapabilityRouteSelection &left,
                         const FuCapabilityRouteSelection &right) {
    return left.selectorNodeOrdinal == right.selectorNodeOrdinal &&
           left.selectedPort == right.selectedPort;
  }
};

/// One correlated FU capability row. Operation and selector ordinals are
/// owner-local physical-node references, never persistent entities or PnR
/// dense IDs.
struct FuCapabilityTemplateSelection final {
  std::vector<std::uint64_t> activeOperationNodeOrdinals;
  std::vector<FuCapabilityRouteSelection> routes;

  friend bool operator==(const FuCapabilityTemplateSelection &left,
                         const FuCapabilityTemplateSelection &right) {
    return left.activeOperationNodeOrdinals ==
               right.activeOperationNodeOrdinals &&
           left.routes == right.routes;
  }
};

/// The finite normalized relation owned by one fabric.fu. This relation
/// correlates operation activation with static mux/demux routes. Exact actor
/// semantics remain owned by OperationSchema and concrete fabric.op
/// capability; physical bit encoding remains owned by ConfigurationABI.
class FuCapabilityDomainRecord final {
public:
  static llvm::Expected<FuCapabilityDomainRecord>
  create(std::vector<FuCapabilityTemplateSelection> templates);

  static llvm::Expected<FuCapabilityDomainRecord>
  fromCanonical(std::vector<FuCapabilityTemplateSelection> templates);

  llvm::ArrayRef<FuCapabilityTemplateSelection> templates() const {
    return templates_;
  }

private:
  explicit FuCapabilityDomainRecord(
      std::vector<FuCapabilityTemplateSelection> templates)
      : templates_(std::move(templates)) {}

  std::vector<FuCapabilityTemplateSelection> templates_;
};

llvm::Expected<std::vector<std::uint8_t>>
encodeFuCapabilityDomainRecord(const FuCapabilityDomainRecord &record);

/// Strict decoding accepts only the canonical normalized wire form and
/// rejects truncation, trailing bytes, duplicate selectors, and duplicate
/// template rows.
llvm::Expected<FuCapabilityDomainRecord>
decodeFuCapabilityDomainRecord(llvm::ArrayRef<std::uint8_t> bytes);

} // namespace fabric

#endif // LOOM_FABRIC_IR_FUCAPABILITYDOMAIN_H
