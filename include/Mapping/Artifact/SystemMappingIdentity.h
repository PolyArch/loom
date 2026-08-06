#ifndef LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGIDENTITY_H
#define LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGIDENTITY_H

#include "Common/Artifact.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

namespace loom::mapping {

using TransferObligationFamilyKey = ::dataflow::CanonicalProducerTerminalRef;
using OperationServiceObligationFamilyKey =
    std::variant<::dataflow::LogicalMemoryRootOrViewRef,
                 ::dataflow::FenceActorFamilyRef>;
using SystemServiceObligationKey =
    std::variant<TransferObligationFamilyKey,
                 OperationServiceObligationFamilyKey>;

struct CanonicalServiceLegKey final {
  SystemServiceObligationKey obligation;
  ::dataflow::ServiceMemberRef member;
  ::dataflow::StructuralOrdinal ordinal = 0;

  friend bool operator==(const CanonicalServiceLegKey &lhs,
                         const CanonicalServiceLegKey &rhs) {
    return lhs.obligation == rhs.obligation && lhs.member == rhs.member &&
           lhs.ordinal == rhs.ordinal;
  }
  friend bool operator!=(const CanonicalServiceLegKey &lhs,
                         const CanonicalServiceLegKey &rhs) {
    return !(lhs == rhs);
  }
};

struct SystemTransferSourceTerminalKey final {
  CanonicalServiceLegKey leg;

  friend bool operator==(const SystemTransferSourceTerminalKey &lhs,
                         const SystemTransferSourceTerminalKey &rhs) {
    return lhs.leg == rhs.leg;
  }
};

struct SystemTransferSinkTerminalKey final {
  CanonicalServiceLegKey leg;
  ::dataflow::StructuralOrdinal sinkOrdinal = 0;

  friend bool operator==(const SystemTransferSinkTerminalKey &lhs,
                         const SystemTransferSinkTerminalKey &rhs) {
    return lhs.leg == rhs.leg && lhs.sinkOrdinal == rhs.sinkOrdinal;
  }
};

using SystemTransferTerminalKey =
    std::variant<SystemTransferSourceTerminalKey,
                 SystemTransferSinkTerminalKey>;

struct DecodedSystemTransferTerminalKeyPrefix final {
  SystemTransferTerminalKey key;
  std::size_t byteCount = 0;
};

llvm::Expected<std::vector<std::uint8_t>>
encodeSystemServiceObligationKey(const ArtifactIdentity &dataflowIdentity,
                                 const SystemServiceObligationKey &key);

llvm::Expected<SystemServiceObligationKey>
decodeSystemServiceObligationKey(llvm::ArrayRef<std::uint8_t> bytes,
                                 const ArtifactIdentity &dataflowIdentity);

llvm::Expected<std::vector<std::uint8_t>>
encodeCanonicalServiceLegKey(const ArtifactIdentity &dataflowIdentity,
                             const CanonicalServiceLegKey &key);

llvm::Expected<CanonicalServiceLegKey>
decodeCanonicalServiceLegKey(llvm::ArrayRef<std::uint8_t> bytes,
                             const ArtifactIdentity &dataflowIdentity);

llvm::Expected<std::vector<std::uint8_t>>
encodeSystemTransferTerminalKey(const ArtifactIdentity &dataflowIdentity,
                                const SystemTransferTerminalKey &key);

llvm::Expected<DecodedSystemTransferTerminalKeyPrefix>
decodeSystemTransferTerminalKeyPrefix(
    llvm::ArrayRef<std::uint8_t> bytes,
    const ArtifactIdentity &dataflowIdentity);

llvm::Expected<SystemTransferTerminalKey>
decodeSystemTransferTerminalKey(llvm::ArrayRef<std::uint8_t> bytes,
                                const ArtifactIdentity &dataflowIdentity);

/// Exact nonpersistent service closure for one Canonical Dataflow Program and
/// canonical root-launch scope. Members, sinks, exposures, and legs are
/// derived from their Dataflow and Canonical Service Schema owners; this view
/// does not select a Fabric target or become Mapping identity.
struct SystemServiceObligationProjection final {
  SystemServiceObligationKey key;
  std::vector<::dataflow::ServiceMemberRef> members;
  std::vector<::dataflow::CanonicalSinkTerminalRef> sinks;
  std::vector<::dataflow::MemoryExposureRef> exposures;
  std::vector<CanonicalServiceLegKey> legs;
};

llvm::Expected<std::vector<SystemServiceObligationProjection>>
projectSystemServiceObligations(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> rootThreadLaunches);

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGIDENTITY_H
