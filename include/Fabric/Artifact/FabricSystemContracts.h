#ifndef LOOM_FABRIC_ARTIFACT_FABRICSYSTEMCONTRACTS_H
#define LOOM_FABRIC_ARTIFACT_FABRICSYSTEMCONTRACTS_H

#include "Fabric/IR/ResourceContract.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace loom {
namespace fabric {

/// Compact target stored by a System field whose dependency role is
/// statically ImportedModule. The dependency table owns the Artifact root;
/// this record owns only its ordinal and the Module-owned local target.
struct FabricImportedModuleTargetRef {
  std::uint64_t dependencyOrdinal = 0;
  FabricModuleTemplateRef target;

  friend bool operator==(const FabricImportedModuleTargetRef &lhs,
                         const FabricImportedModuleTargetRef &rhs) {
    return lhs.dependencyOrdinal == rhs.dependencyOrdinal &&
           lhs.target == rhs.target;
  }
};

std::vector<std::uint8_t> encodeFabricImportedModuleTargetRef(
    const FabricImportedModuleTargetRef &reference);

llvm::Expected<FabricImportedModuleTargetRef>
decodeFabricImportedModuleTargetRef(llvm::ArrayRef<std::uint8_t> bytes);

/// One Module boundary endpoint scoped by the System root's canonical
/// ImportedModule dependency table. This is an attachment field, not a new
/// artifact-local reference family.
struct FabricImportedModuleBoundaryEndpointRef {
  std::uint64_t dependencyOrdinal = 0;
  FabricModuleBoundaryEndpointRef target;

  friend bool operator==(const FabricImportedModuleBoundaryEndpointRef &lhs,
                         const FabricImportedModuleBoundaryEndpointRef &rhs) {
    return lhs.dependencyOrdinal == rhs.dependencyOrdinal &&
           lhs.target == rhs.target;
  }
};

std::vector<std::uint8_t> encodeFabricImportedModuleBoundaryEndpointRef(
    const FabricImportedModuleBoundaryEndpointRef &reference);

llvm::Expected<FabricImportedModuleBoundaryEndpointRef>
decodeFabricImportedModuleBoundaryEndpointRef(
    llvm::ArrayRef<std::uint8_t> bytes);

/// The occurrence-side endpoint of a module-to-SpatialCore attachment. The
/// selected plane is explicit while owner and ordinal remain the existing
/// transport or memory endpoint reference. Only a SpatialCore occurrence may
/// own this field.
class FabricSpatialAttachmentEndpointRef {
public:
  enum class Plane : std::uint32_t { Transport, Memory };

  static llvm::Expected<FabricSpatialAttachmentEndpointRef>
  create(FabricTransportEndpointRef endpoint);
  static llvm::Expected<FabricSpatialAttachmentEndpointRef>
  create(FabricMemoryEndpointRef endpoint);

  Plane plane() const {
    return std::holds_alternative<FabricTransportEndpointRef>(endpoint_)
               ? Plane::Transport
               : Plane::Memory;
  }
  const FabricTransportEndpointRef *transport() const {
    return std::get_if<FabricTransportEndpointRef>(&endpoint_);
  }
  const FabricMemoryEndpointRef *memory() const {
    return std::get_if<FabricMemoryEndpointRef>(&endpoint_);
  }

  friend bool operator==(const FabricSpatialAttachmentEndpointRef &lhs,
                         const FabricSpatialAttachmentEndpointRef &rhs) {
    return lhs.endpoint_ == rhs.endpoint_;
  }

private:
  using Endpoint =
      std::variant<FabricTransportEndpointRef, FabricMemoryEndpointRef>;

  explicit FabricSpatialAttachmentEndpointRef(Endpoint endpoint)
      : endpoint_(std::move(endpoint)) {}

  Endpoint endpoint_;
};

std::vector<std::uint8_t> encodeFabricSpatialAttachmentEndpointRef(
    const FabricSpatialAttachmentEndpointRef &reference);

llvm::Expected<FabricSpatialAttachmentEndpointRef>
decodeFabricSpatialAttachmentEndpointRef(llvm::ArrayRef<std::uint8_t> bytes);

/// One architecture-visible traversal mode of a System transport resource.
/// The selected UsePattern owns the atomic claims, eligibility, timing,
/// progress, and arbitration semantics. This record owns only topology and
/// the exact relation to that existing atomic use.
class SystemTransferPatternRecord {
public:
  static llvm::Expected<SystemTransferPatternRecord>
  create(FabricTransferPatternRef pattern, FabricTransportEndpointRef ingress,
         std::vector<FabricTransportEndpointRef> egresses,
         FabricUsePatternRef usePattern);

  const FabricTransferPatternRef &pattern() const { return pattern_; }
  const FabricTransportEndpointRef &ingress() const { return ingress_; }
  llvm::ArrayRef<FabricTransportEndpointRef> egresses() const {
    return egresses_;
  }
  const FabricUsePatternRef &usePattern() const { return usePattern_; }

  friend bool operator==(const SystemTransferPatternRecord &lhs,
                         const SystemTransferPatternRecord &rhs) {
    return lhs.pattern_ == rhs.pattern_ && lhs.ingress_ == rhs.ingress_ &&
           lhs.egresses_ == rhs.egresses_ && lhs.usePattern_ == rhs.usePattern_;
  }

private:
  SystemTransferPatternRecord(FabricTransferPatternRef pattern,
                              FabricTransportEndpointRef ingress,
                              std::vector<FabricTransportEndpointRef> egresses,
                              FabricUsePatternRef usePattern)
      : pattern_(std::move(pattern)), ingress_(std::move(ingress)),
        egresses_(std::move(egresses)), usePattern_(std::move(usePattern)) {}

  FabricTransferPatternRef pattern_;
  FabricTransportEndpointRef ingress_;
  std::vector<FabricTransportEndpointRef> egresses_;
  FabricUsePatternRef usePattern_;
};

std::vector<std::uint8_t>
encodeSystemTransferPatternRecord(const SystemTransferPatternRecord &record);

llvm::Expected<SystemTransferPatternRecord>
decodeSystemTransferPatternRecord(llvm::ArrayRef<std::uint8_t> bytes);

enum class RiscVXLen : std::uint32_t { X32, X64 };
enum class RiscVBase : std::uint32_t { I, E };
enum class RiscVExtension : std::uint32_t {
  M,
  A,
  F,
  D,
  C,
  V,
  Zicsr,
  Zifencei,
  Zba,
  Zbb,
  Zbs,
  Ztso,
};
enum class InstructionEndianness : std::uint32_t { Little, Big };
enum class PrivilegeMode : std::uint32_t { User, Supervisor, Machine };
enum class RiscVAbi : std::uint32_t {
  Ilp32,
  Ilp32e,
  Ilp32f,
  Ilp32d,
  Lp64,
  Lp64f,
  Lp64d,
};
enum class RiscVMemoryOrdering : std::uint32_t { Rvwmo, Ztso };
enum class InstructionSyncScope : std::uint32_t {
  SingleThread,
  Hart,
  System,
};
enum class RiscVCodeModel : std::uint32_t { MediumLow, MediumAny };
enum class RelocationModel : std::uint32_t {
  Static,
  PositionIndependent,
};
enum class InstructionRuntimeService : std::uint32_t {
  ThreadDispatch,
  SpatialLaunch,
  MemoryAllocation,
  AtomicRuntime,
};

struct RiscVArchitectureDeclaration {
  RiscVXLen xlen = RiscVXLen::X32;
  RiscVBase base = RiscVBase::I;
  std::vector<RiscVExtension> extensions;
  InstructionEndianness endianness = InstructionEndianness::Little;
  std::uint32_t physicalAddressWidthBits = 0;
  std::vector<PrivilegeMode> privilegeModes;
  std::vector<RiscVAbi> abiCapabilities;
  RiscVMemoryOrdering memoryOrdering = RiscVMemoryOrdering::Rvwmo;
  std::vector<InstructionSyncScope> syncScopes;
  std::vector<RiscVCodeModel> codeModels;
  std::vector<RelocationModel> relocationModels;
  std::vector<InstructionRuntimeService> runtimeServices;
};

/// The closed binary-compatibility contract of one InstructionCore. The
/// constructor normalizes authoring set order and rejects duplicate or
/// inconsistent entries. Strict persistent import additionally requires the
/// input bytes to equal the canonical re-encoding.
class InstructionCoreArchitecturalContract {
public:
  static llvm::Expected<InstructionCoreArchitecturalContract>
  create(RiscVArchitectureDeclaration declaration);

  RiscVXLen xlen() const { return declaration_.xlen; }
  RiscVBase base() const { return declaration_.base; }
  llvm::ArrayRef<RiscVExtension> extensions() const {
    return declaration_.extensions;
  }
  InstructionEndianness endianness() const { return declaration_.endianness; }
  std::uint32_t physicalAddressWidthBits() const {
    return declaration_.physicalAddressWidthBits;
  }
  llvm::ArrayRef<PrivilegeMode> privilegeModes() const {
    return declaration_.privilegeModes;
  }
  llvm::ArrayRef<RiscVAbi> abiCapabilities() const {
    return declaration_.abiCapabilities;
  }
  RiscVMemoryOrdering memoryOrdering() const {
    return declaration_.memoryOrdering;
  }
  llvm::ArrayRef<InstructionSyncScope> syncScopes() const {
    return declaration_.syncScopes;
  }
  llvm::ArrayRef<RiscVCodeModel> codeModels() const {
    return declaration_.codeModels;
  }
  llvm::ArrayRef<RelocationModel> relocationModels() const {
    return declaration_.relocationModels;
  }
  llvm::ArrayRef<InstructionRuntimeService> runtimeServices() const {
    return declaration_.runtimeServices;
  }

private:
  explicit InstructionCoreArchitecturalContract(
      RiscVArchitectureDeclaration declaration)
      : declaration_(std::move(declaration)) {}

  RiscVArchitectureDeclaration declaration_;
};

enum class InstructionOperationClass : std::uint32_t {
  IntegerAlu,
  IntegerMultiply,
  IntegerDivide,
  Branch,
  LoadStore,
  FloatingPointAlu,
  FloatingPointMultiply,
  FloatingPointDivide,
  VectorAlu,
  VectorMultiply,
  System,
};

struct ExecutionUnitRecord {
  InstructionOperationClass operationClass;
  std::uint32_t count;
  std::uint32_t latencyCycles;
  std::uint32_t initiationInterval;
};

struct InstructionCoreCommonDeclaration {
  std::uint32_t hardwareThreadCount;
  std::vector<ExecutionUnitRecord> executionUnits;
  ::fabric::ResourceContract resourceContract;
};

struct InOrderMicroarchitectureDeclaration {
  std::uint32_t fetchWidth;
  std::uint32_t decodeWidth;
  std::uint32_t issueWidth;
  std::uint32_t commitWidth;
  std::uint32_t memoryIssueWidth;
  std::uint32_t memoryCommitWidth;
  std::uint32_t maxOutstandingMemoryOperations;
  std::uint32_t storeBufferEntries;
};

struct OutOfOrderMicroarchitectureDeclaration {
  std::uint32_t fetchWidth;
  std::uint32_t decodeWidth;
  std::uint32_t renameWidth;
  std::uint32_t dispatchWidth;
  std::uint32_t issueWidth;
  std::uint32_t writebackWidth;
  std::uint32_t commitWidth;
  std::uint32_t reorderBufferEntries;
  std::uint32_t issueQueueEntries;
  std::uint32_t loadQueueEntries;
  std::uint32_t storeQueueEntries;
  std::uint32_t physicalIntegerRegisters;
  std::uint32_t physicalFloatRegisters;
  std::uint32_t physicalVectorRegisters;
};

enum class InstructionCoreRealizationKind : std::uint32_t {
  InOrder,
  OutOfOrder,
};

/// The closed timing and capacity realization of one InstructionCore. Its
/// embedded ResourceContract is the only Mapping-visible resource authority.
class InstructionCoreMicroarchitecturalRealization {
public:
  static llvm::Expected<InstructionCoreMicroarchitecturalRealization>
  createInOrder(InstructionCoreCommonDeclaration common,
                InOrderMicroarchitectureDeclaration pipeline);
  static llvm::Expected<InstructionCoreMicroarchitecturalRealization>
  createOutOfOrder(InstructionCoreCommonDeclaration common,
                   OutOfOrderMicroarchitectureDeclaration pipeline);

  InstructionCoreRealizationKind kind() const { return kind_; }
  std::uint32_t hardwareThreadCount() const { return hardwareThreadCount_; }
  llvm::ArrayRef<ExecutionUnitRecord> executionUnits() const {
    return executionUnits_;
  }
  const ::fabric::ResourceContract &resourceContract() const {
    return resourceContract_;
  }
  const InOrderMicroarchitectureDeclaration *inOrder() const {
    return std::get_if<InOrderMicroarchitectureDeclaration>(&pipeline_);
  }
  const OutOfOrderMicroarchitectureDeclaration *outOfOrder() const {
    return std::get_if<OutOfOrderMicroarchitectureDeclaration>(&pipeline_);
  }

private:
  using Pipeline = std::variant<InOrderMicroarchitectureDeclaration,
                                OutOfOrderMicroarchitectureDeclaration>;

  InstructionCoreMicroarchitecturalRealization(
      InstructionCoreRealizationKind kind, std::uint32_t hardwareThreadCount,
      std::vector<ExecutionUnitRecord> executionUnits,
      ::fabric::ResourceContract resourceContract, Pipeline pipeline)
      : kind_(kind), hardwareThreadCount_(hardwareThreadCount),
        executionUnits_(std::move(executionUnits)),
        resourceContract_(std::move(resourceContract)),
        pipeline_(std::move(pipeline)) {}

  InstructionCoreRealizationKind kind_;
  std::uint32_t hardwareThreadCount_;
  std::vector<ExecutionUnitRecord> executionUnits_;
  ::fabric::ResourceContract resourceContract_;
  Pipeline pipeline_;
};

llvm::Expected<std::vector<std::uint8_t>>
encodeInstructionCoreArchitecturalContract(
    const InstructionCoreArchitecturalContract &contract);
llvm::Expected<InstructionCoreArchitecturalContract>
decodeInstructionCoreArchitecturalContract(llvm::ArrayRef<std::uint8_t> bytes);

llvm::Expected<std::vector<std::uint8_t>>
encodeInstructionCoreMicroarchitecturalRealization(
    const InstructionCoreMicroarchitecturalRealization &realization);
llvm::Expected<InstructionCoreMicroarchitecturalRealization>
decodeInstructionCoreMicroarchitecturalRealization(
    llvm::ArrayRef<std::uint8_t> bytes);

enum class ResetPolarity : std::uint32_t {
  ActiveHigh,
  ActiveLow,
};

enum class ResetTiming : std::uint32_t {
  Synchronous,
  Asynchronous,
};

enum class ResetInitialState : std::uint32_t {
  Asserted,
  Deasserted,
};

class ClockDomainContractRecord {
public:
  static llvm::Expected<ClockDomainContractRecord>
  create(std::uint64_t periodFs, std::uint64_t phaseFs);

  std::uint64_t periodFs() const { return periodFs_; }
  std::uint64_t phaseFs() const { return phaseFs_; }

  friend bool operator==(const ClockDomainContractRecord &lhs,
                         const ClockDomainContractRecord &rhs) {
    return lhs.periodFs_ == rhs.periodFs_ && lhs.phaseFs_ == rhs.phaseFs_;
  }
  friend bool operator!=(const ClockDomainContractRecord &lhs,
                         const ClockDomainContractRecord &rhs) {
    return !(lhs == rhs);
  }

private:
  ClockDomainContractRecord(std::uint64_t periodFs, std::uint64_t phaseFs)
      : periodFs_(periodFs), phaseFs_(phaseFs) {}

  std::uint64_t periodFs_;
  std::uint64_t phaseFs_;
};

class ResetDomainContractRecord {
public:
  static llvm::Expected<ResetDomainContractRecord>
  create(ResetPolarity polarity, ResetTiming assertion, ResetTiming deassertion,
         ResetInitialState initialState,
         std::optional<ClockDomainRef> synchronousTo,
         std::uint32_t releaseLatencyCycles);

  ResetPolarity polarity() const { return polarity_; }
  ResetTiming assertion() const { return assertion_; }
  ResetTiming deassertion() const { return deassertion_; }
  ResetInitialState initialState() const { return initialState_; }
  const std::optional<ClockDomainRef> &synchronousTo() const {
    return synchronousTo_;
  }
  std::uint32_t releaseLatencyCycles() const { return releaseLatencyCycles_; }

  friend bool operator==(const ResetDomainContractRecord &lhs,
                         const ResetDomainContractRecord &rhs) {
    return lhs.polarity_ == rhs.polarity_ && lhs.assertion_ == rhs.assertion_ &&
           lhs.deassertion_ == rhs.deassertion_ &&
           lhs.initialState_ == rhs.initialState_ &&
           lhs.synchronousTo_ == rhs.synchronousTo_ &&
           lhs.releaseLatencyCycles_ == rhs.releaseLatencyCycles_;
  }
  friend bool operator!=(const ResetDomainContractRecord &lhs,
                         const ResetDomainContractRecord &rhs) {
    return !(lhs == rhs);
  }

private:
  ResetDomainContractRecord(ResetPolarity polarity, ResetTiming assertion,
                            ResetTiming deassertion,
                            ResetInitialState initialState,
                            std::optional<ClockDomainRef> synchronousTo,
                            std::uint32_t releaseLatencyCycles)
      : polarity_(polarity), assertion_(assertion), deassertion_(deassertion),
        initialState_(initialState), synchronousTo_(std::move(synchronousTo)),
        releaseLatencyCycles_(releaseLatencyCycles) {}

  ResetPolarity polarity_;
  ResetTiming assertion_;
  ResetTiming deassertion_;
  ResetInitialState initialState_;
  std::optional<ClockDomainRef> synchronousTo_;
  std::uint32_t releaseLatencyCycles_;
};

/// Version 1 has one crossing variant. Depth and synchronizer stages are
/// canonical unsigned 32-bit fields; the carrier remains structural context.
class ClockCrossingContractRecord {
public:
  static llvm::Expected<ClockCrossingContractRecord>
  createAsyncFifo(FabricTransferPatternRef transferPattern,
                  ClockDomainRef sourceClock, ClockDomainRef destinationClock,
                  std::uint32_t depth, std::uint32_t synchronizerStages);

  const FabricTransferPatternRef &transferPattern() const {
    return transferPattern_;
  }
  const ClockDomainRef &sourceClock() const { return sourceClock_; }
  const ClockDomainRef &destinationClock() const { return destinationClock_; }
  std::uint32_t depth() const { return depth_; }
  std::uint32_t synchronizerStages() const { return synchronizerStages_; }

  friend bool operator==(const ClockCrossingContractRecord &lhs,
                         const ClockCrossingContractRecord &rhs) {
    return lhs.transferPattern_ == rhs.transferPattern_ &&
           lhs.sourceClock_ == rhs.sourceClock_ &&
           lhs.destinationClock_ == rhs.destinationClock_ &&
           lhs.depth_ == rhs.depth_ &&
           lhs.synchronizerStages_ == rhs.synchronizerStages_;
  }
  friend bool operator!=(const ClockCrossingContractRecord &lhs,
                         const ClockCrossingContractRecord &rhs) {
    return !(lhs == rhs);
  }

private:
  ClockCrossingContractRecord(FabricTransferPatternRef transferPattern,
                              ClockDomainRef sourceClock,
                              ClockDomainRef destinationClock,
                              std::uint32_t depth,
                              std::uint32_t synchronizerStages)
      : transferPattern_(std::move(transferPattern)),
        sourceClock_(std::move(sourceClock)),
        destinationClock_(std::move(destinationClock)), depth_(depth),
        synchronizerStages_(synchronizerStages) {}

  FabricTransferPatternRef transferPattern_;
  ClockDomainRef sourceClock_;
  ClockDomainRef destinationClock_;
  std::uint32_t depth_;
  std::uint32_t synchronizerStages_;
};

/// These codecs own only the canonical record bytes and intrinsic field
/// invariants. Exact artifact scope, refined domain kinds, transfer-pattern
/// ownership, and whole-root relations are validated by the sealed System root
/// importer that embeds these records.
llvm::Expected<std::vector<std::uint8_t>>
encodeClockDomainContractRecord(const ClockDomainContractRecord &record);
llvm::Expected<ClockDomainContractRecord>
decodeClockDomainContractRecord(llvm::ArrayRef<std::uint8_t> bytes);

llvm::Expected<std::vector<std::uint8_t>>
encodeResetDomainContractRecord(const ResetDomainContractRecord &record);
llvm::Expected<ResetDomainContractRecord>
decodeResetDomainContractRecord(llvm::ArrayRef<std::uint8_t> bytes);

llvm::Expected<std::vector<std::uint8_t>>
encodeClockCrossingContractRecord(const ClockCrossingContractRecord &record);
llvm::Expected<ClockCrossingContractRecord>
decodeClockCrossingContractRecord(llvm::ArrayRef<std::uint8_t> bytes);

} // namespace fabric
} // namespace loom

#endif // LOOM_FABRIC_ARTIFACT_FABRICSYSTEMCONTRACTS_H
