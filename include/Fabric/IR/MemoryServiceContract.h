#ifndef LOOM_FABRIC_IR_MEMORY_SERVICE_CONTRACT_H
#define LOOM_FABRIC_IR_MEMORY_SERVICE_CONTRACT_H

#include "Fabric/IR/MemoryActorContractDomain.h"
#include "Fabric/IR/MemoryCapabilityDomains.h"
#include "Fabric/IR/MemoryConsistencyContract.h"
#include "Fabric/IR/ResourceContract.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace fabric {

enum class MemoryServiceOwnerKind : std::uint32_t { Local, System };

enum class MemoryServiceRegionBehavior : std::uint32_t { Storage, Mmio };

struct MemoryServiceRegionDeclaration {
  std::uint64_t addressBaseBytes = 0;
  std::uint64_t sizeBytes = 0;
  MemoryServiceRegionBehavior behavior = MemoryServiceRegionBehavior::Storage;
  std::optional<ParameterizedMemoryAccessDomain> mmioAcceptedAccessDomain;
};

struct NoMemoryServiceConsistency {};

inline bool operator==(NoMemoryServiceConsistency, NoMemoryServiceConsistency) {
  return true;
}

struct LocalBoundedCompletionCycles {
  std::uint64_t maxIssueToRetireCycles = 0;
};

inline bool operator==(LocalBoundedCompletionCycles left,
                       LocalBoundedCompletionCycles right) {
  return left.maxIssueToRetireCycles == right.maxIssueToRetireCycles;
}

using LocalProviderProgress =
    std::variant<LocalBoundedCompletionCycles, FairEventual>;

struct LocalProviderConsistency {
  ReleaseVisibilityPoint releaseVisibilityPoint =
      ReleaseVisibilityPoint::AtLinearization;
  LocalProviderProgress progress;
};

inline bool operator==(const LocalProviderConsistency &left,
                       const LocalProviderConsistency &right) {
  return left.releaseVisibilityPoint == right.releaseVisibilityPoint &&
         left.progress == right.progress;
}

using MemoryServiceConsistencyBinding =
    std::variant<NoMemoryServiceConsistency, LocalProviderConsistency,
                 loom::fabric::MemoryConsistencyDomainRef>;

struct MemoryServiceCapabilityDeclaration {
  MemoryActorContractDomain actorContractDomain;
  std::optional<ParameterizedMemoryAccessDomain> accessDomain;
  std::vector<std::uint64_t> serviceRegionOrdinals;
  std::uint64_t serviceBeatWidthBits = 0;
  std::vector<UsePatternKey> admissibleUsePatterns;
  MemoryServiceConsistencyBinding consistencyBinding;
};

struct MemoryServiceContractDeclaration {
  std::vector<MemoryServiceRegionDeclaration> regions;
  ResourceContract resourceContract;
  std::vector<MemoryServiceCapabilityDeclaration> capabilities;
};

/// One complete canonical memory-service capability contract shared by local
/// fabric.mem services and System memory services. Owner kind is validation
/// context and is not repeated in the persistent record.
class MemoryServiceContractRecord {
public:
  static llvm::Expected<MemoryServiceContractRecord>
  create(mlir::MLIRContext *context, MemoryServiceOwnerKind owner,
         MemoryServiceContractDeclaration declaration);

  static llvm::Expected<MemoryServiceContractRecord>
  fromCanonical(mlir::MLIRContext *context, MemoryServiceOwnerKind owner,
                MemoryServiceContractDeclaration declaration);

  llvm::ArrayRef<MemoryServiceRegionDeclaration> regions() const {
    return declaration_.regions;
  }
  const ResourceContract &resourceContract() const {
    return declaration_.resourceContract;
  }
  llvm::ArrayRef<MemoryServiceCapabilityDeclaration> capabilities() const {
    return declaration_.capabilities;
  }

private:
  explicit MemoryServiceContractRecord(
      MemoryServiceContractDeclaration declaration)
      : declaration_(std::move(declaration)) {}

  MemoryServiceContractDeclaration declaration_;
};

llvm::Expected<std::vector<std::uint8_t>>
encodeMemoryServiceContractRecord(const MemoryServiceContractRecord &record);

llvm::Expected<MemoryServiceContractRecord>
decodeMemoryServiceContractRecord(llvm::ArrayRef<std::uint8_t> bytes,
                                  mlir::MLIRContext *context,
                                  MemoryServiceOwnerKind owner);

/// A local service's regions are relative offsets and must fit the occurrence
/// capacity owned by LocalMemoryServiceAttr.
llvm::Error
validateLocalMemoryServiceCapacity(const MemoryServiceContractRecord &record,
                                   std::uint64_t capacityBytes);

} // namespace fabric

#endif // LOOM_FABRIC_IR_MEMORY_SERVICE_CONTRACT_H
