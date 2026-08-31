#ifndef LOOM_SIMULATOR_CGRACLOSEDWAITCERTIFICATE_H
#define LOOM_SIMULATOR_CGRACLOSEDWAITCERTIFICATE_H

#include "Simulator/CGRASimulator.h"

#include "Common/ComponentViewDigest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace loom::sim {

/// Minimal persistent transfer fact referenced by one closed-wait edge. It is
/// deliberately distinct from the larger invocation diagnostic record: every
/// field here is encoded, decoded, and independently consumed.
struct CgraClosedWaitTransfer final {
  CgraClosedWaitTransfer(
      std::uint64_t binding, std::uint64_t occurrence, llvm::APInt tag,
      bool isTagged,
      ::dataflow::CanonicalGraphProducerEndpointRef logicalProducer)
      : bindingOrdinal(binding), occurrenceOrdinal(occurrence),
        physicalTagValue(std::move(tag)), tagged(isTagged),
        producer(std::move(logicalProducer)) {}

  std::uint64_t bindingOrdinal;
  std::uint64_t occurrenceOrdinal;
  llvm::APInt physicalTagValue;
  bool tagged = false;
  ::dataflow::CanonicalGraphProducerEndpointRef producer;
  std::optional<CgraPhysicalTagMappingOwner> physicalTagOwner;
  std::uint64_t blockingStorageOrdinal = 0;
  std::uint64_t blockingDownstreamStorageOrdinal = 0;
  std::vector<::loom::fabric::FabricPhysicalTraversalRef> blockingTraversals;
  std::vector<::loom::fabric::FabricPhysicalTraversalRef>
      blockingDownstreamTraversals;
};

/// Durable semantic core of one runtime-proven closed wait. The invocation
/// diagnostic remains a removable observation projection; this value contains
/// only the owner closure, exact dynamic transfers, and the closed SCC edges
/// needed by independent consumers.
struct CgraClosedWaitCertificate final {
  explicit CgraClosedWaitCertificate(CgraExecutionOwnerReferences ownerRefs)
      : owners(std::move(ownerRefs)) {}

  CgraExecutionOwnerReferences owners;
  std::vector<CgraClosedWaitTransfer> transfers;
  std::vector<CgraClosedWaitSetDiagnostic::WaitEdge> edges;
};

/// Extracts and canonicalizes a durable certificate from an invocation-local
/// diagnostic. Missing owners, proof failure, an open relation, duplicate
/// transfers, or an edge whose transfer cannot be recovered fails closed.
llvm::Expected<CgraClosedWaitCertificate>
buildCgraClosedWaitCertificate(
    const CgraClosedWaitSetDiagnostic &diagnostic);

/// Independently verifies the complete typed certificate.
llvm::Error
verifyCgraClosedWaitCertificate(const CgraClosedWaitCertificate &certificate);

/// Canonical owner wire used by the Evaluation terminal-witness codec.
llvm::Expected<std::vector<std::uint8_t>>
encodeCgraClosedWaitCertificate(
    const CgraClosedWaitCertificate &certificate);
llvm::Expected<CgraClosedWaitCertificate>
decodeCgraClosedWaitCertificate(llvm::ArrayRef<std::uint8_t> bytes);

inline constexpr llvm::StringLiteral cgraClosedWaitCertificateDigestDomain =
    "loom.cgra_closed_wait_certificate.1";
inline constexpr llvm::StringLiteral cgraClosedWaitStructureDigestDomain =
    "loom.cgra_closed_wait_structure.1";

/// Domain-separated digest of the complete canonical certificate wire. This
/// type cannot be confused with a Mapping, config-view, or diagnostic digest.
class CgraClosedWaitCertificateDigest final {
public:
  const ComponentViewDigest &value() const { return value_; }

  friend bool operator==(const CgraClosedWaitCertificateDigest &lhs,
                         const CgraClosedWaitCertificateDigest &rhs) {
    return lhs.value_ == rhs.value_;
  }
  friend bool operator!=(const CgraClosedWaitCertificateDigest &lhs,
                         const CgraClosedWaitCertificateDigest &rhs) {
    return !(lhs == rhs);
  }

private:
  explicit CgraClosedWaitCertificateDigest(ComponentViewDigest value)
      : value_(std::move(value)) {}

  ComponentViewDigest value_;

  friend llvm::Expected<CgraClosedWaitCertificateDigest>
  digestCgraClosedWaitCertificate(const CgraClosedWaitCertificate &);
};

llvm::Expected<CgraClosedWaitCertificateDigest>
digestCgraClosedWaitCertificate(
    const CgraClosedWaitCertificate &certificate);

std::string formatCgraClosedWaitCertificateDigest(
    const CgraClosedWaitCertificateDigest &digest);

/// Request-independent grouping key for the complete dynamic wait structure.
/// It retains exact D/T/F owners, transfers, queue classes, tags, storage
/// owners, occurrences, and wait edges, but deliberately omits the
/// SpatialMapping root. The full certificate digest remains the durable
/// Evidence identity. Structural equality across different Mappings is
/// diagnostic only: an exact-assignment no-good may still soundly exclude the
/// new Mapping and continue cumulative repair.
class CgraClosedWaitStructureDigest final {
public:
  const ComponentViewDigest &value() const { return value_; }

  friend bool operator==(const CgraClosedWaitStructureDigest &lhs,
                         const CgraClosedWaitStructureDigest &rhs) {
    return lhs.value_ == rhs.value_;
  }
  friend bool operator!=(const CgraClosedWaitStructureDigest &lhs,
                         const CgraClosedWaitStructureDigest &rhs) {
    return !(lhs == rhs);
  }

private:
  explicit CgraClosedWaitStructureDigest(ComponentViewDigest value)
      : value_(std::move(value)) {}

  ComponentViewDigest value_;

  friend llvm::Expected<CgraClosedWaitStructureDigest>
  digestCgraClosedWaitStructure(const CgraClosedWaitCertificate &);
};

llvm::Expected<CgraClosedWaitStructureDigest>
digestCgraClosedWaitStructure(const CgraClosedWaitCertificate &certificate);

std::string formatCgraClosedWaitStructureDigest(
    const CgraClosedWaitStructureDigest &digest);

} // namespace loom::sim

#endif // LOOM_SIMULATOR_CGRACLOSEDWAITCERTIFICATE_H
