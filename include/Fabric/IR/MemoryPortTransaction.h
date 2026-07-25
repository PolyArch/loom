#ifndef LOOM_FABRIC_IR_MEMORY_PORT_TRANSACTION_H
#define LOOM_FABRIC_IR_MEMORY_PORT_TRANSACTION_H

#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/IR/ResourceContract.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace fabric {

enum class MemoryPortTransactionProjection : std::uint8_t {
  Direct,
  ActiveLanesRowMajor,
};

std::uint8_t
getCanonicalTag(MemoryPortTransactionProjection transactionProjection);

llvm::Expected<MemoryPortTransactionProjection>
decodeMemoryPortTransactionProjection(std::uint8_t tag);

class MemoryOperationPortResourceView;

class MemoryOperationPatternView {
public:
  const MemoryOperationPortResourceView &operationPort() const {
    return *operationPort_;
  }
  const loom::fabric::FabricUsePatternRef &usePatternRef() const {
    return usePattern_;
  }
  UsePattern usePattern() const;
  MemoryPortTransactionProjection transactionProjection() const {
    return transactionProjection_;
  }

private:
  MemoryOperationPatternView(
      const MemoryOperationPortResourceView &operationPort,
      loom::fabric::FabricUsePatternRef usePattern,
      MemoryPortTransactionProjection transactionProjection)
      : operationPort_(&operationPort), usePattern_(std::move(usePattern)),
        transactionProjection_(transactionProjection) {}

  const MemoryOperationPortResourceView *operationPort_;
  loom::fabric::FabricUsePatternRef usePattern_;
  MemoryPortTransactionProjection transactionProjection_;

  friend class MemoryOperationPortResourceView;
};

/// The complete validated ResourceContract embedded by one memory-operation
/// port. State and use-pattern references are derived from this exact owner;
/// no count or ownerless ordinal is accepted as an alternate interface.
class MemoryOperationPortResourceView {
public:
  static llvm::Expected<MemoryOperationPortResourceView> create(
      loom::fabric::FabricMemoryOperationPortRef owner,
      ResourceContract resourceContract,
      llvm::ArrayRef<MemoryPortTransactionProjection> transactionProjections);

  const loom::fabric::FabricMemoryOperationPortRef &owner() const {
    return owner_;
  }
  const ResourceContract &resourceContract() const { return resourceContract_; }
  llvm::ArrayRef<loom::fabric::FabricResourceStateRef> resourceStates() const {
    return resourceStates_;
  }
  llvm::ArrayRef<loom::fabric::FabricUsePatternRef> usePatterns() const {
    return usePatterns_;
  }

  llvm::Expected<MemoryOperationPatternView>
  operationPattern(const loom::fabric::FabricUsePatternRef &usePattern) const;

private:
  MemoryOperationPortResourceView(
      loom::fabric::FabricMemoryOperationPortRef owner,
      ResourceContract resourceContract,
      std::vector<loom::fabric::FabricResourceStateRef> resourceStates,
      std::vector<loom::fabric::FabricUsePatternRef> usePatterns,
      std::vector<MemoryPortTransactionProjection> transactionProjections)
      : owner_(std::move(owner)),
        resourceContract_(std::move(resourceContract)),
        resourceStates_(std::move(resourceStates)),
        usePatterns_(std::move(usePatterns)),
        transactionProjections_(std::move(transactionProjections)) {}

  loom::fabric::FabricMemoryOperationPortRef owner_;
  ResourceContract resourceContract_;
  std::vector<loom::fabric::FabricResourceStateRef> resourceStates_;
  std::vector<loom::fabric::FabricUsePatternRef> usePatterns_;
  std::vector<MemoryPortTransactionProjection> transactionProjections_;
};

enum class MemoryChildActivationKind : std::uint8_t {
  Always,
  ParentMaskAny,
  ParentMaskLane,
};

class MemoryChildActivation {
public:
  MemoryChildActivationKind kind() const { return kind_; }
  std::optional<std::uint64_t> lane() const {
    if (kind_ != MemoryChildActivationKind::ParentMaskLane)
      return std::nullopt;
    return lane_;
  }

private:
  MemoryChildActivation(MemoryChildActivationKind kind, std::uint64_t lane)
      : kind_(kind), lane_(lane) {}

  static MemoryChildActivation always() {
    return MemoryChildActivation(MemoryChildActivationKind::Always, 0);
  }
  static MemoryChildActivation parentMaskAny() {
    return MemoryChildActivation(MemoryChildActivationKind::ParentMaskAny, 0);
  }
  static MemoryChildActivation parentMaskLane(std::uint64_t lane) {
    return MemoryChildActivation(MemoryChildActivationKind::ParentMaskLane,
                                 lane);
  }

  MemoryChildActivationKind kind_;
  std::uint64_t lane_;

  friend llvm::Expected<class MemoryPortTransactionPlan>
  deriveMemoryPortTransactionPlan(
      const MemoryOperationPatternView &pattern,
      const dataflow::CanonicalActorSchemaProjection &actor,
      const dataflow::semantics::CanonicalService &parentService,
      const std::optional<dataflow::semantics::CanonicalMemoryAccessView>
          &access);
};

enum class MemoryChildProjectionKind : std::uint8_t {
  ParentRequest,
  ElementLane,
};

class MemoryChildProjection {
public:
  MemoryChildProjectionKind kind() const { return kind_; }
  std::optional<std::uint64_t> lane() const {
    if (kind_ != MemoryChildProjectionKind::ElementLane)
      return std::nullopt;
    return lane_;
  }

private:
  MemoryChildProjection(MemoryChildProjectionKind kind, std::uint64_t lane)
      : kind_(kind), lane_(lane) {}

  static MemoryChildProjection parentRequest() {
    return MemoryChildProjection(MemoryChildProjectionKind::ParentRequest, 0);
  }
  static MemoryChildProjection elementLane(std::uint64_t lane) {
    return MemoryChildProjection(MemoryChildProjectionKind::ElementLane, lane);
  }

  MemoryChildProjectionKind kind_;
  std::uint64_t lane_;

  friend llvm::Expected<class MemoryPortTransactionPlan>
  deriveMemoryPortTransactionPlan(
      const MemoryOperationPatternView &pattern,
      const dataflow::CanonicalActorSchemaProjection &actor,
      const dataflow::semantics::CanonicalService &parentService,
      const std::optional<dataflow::semantics::CanonicalMemoryAccessView>
          &access);
};

class MemoryPortChildTransaction {
public:
  std::uint64_t ordinal() const { return ordinal_; }
  const MemoryChildActivation &activation() const { return activation_; }
  const MemoryChildProjection &projection() const { return projection_; }

private:
  MemoryPortChildTransaction(std::uint64_t ordinal,
                             MemoryChildActivation activation,
                             MemoryChildProjection projection)
      : ordinal_(ordinal), activation_(activation), projection_(projection) {}

  std::uint64_t ordinal_;
  MemoryChildActivation activation_;
  MemoryChildProjection projection_;

  friend llvm::Expected<class MemoryPortTransactionPlan>
  deriveMemoryPortTransactionPlan(
      const MemoryOperationPatternView &pattern,
      const dataflow::CanonicalActorSchemaProjection &actor,
      const dataflow::semantics::CanonicalService &parentService,
      const std::optional<dataflow::semantics::CanonicalMemoryAccessView>
          &access);
};

enum class MemoryResultAssemblyStrategy : std::uint8_t {
  PassThroughParent,
  ParentResponseOrZeroOnEmptyMask,
  RowMajorLaneValues,
};

enum class MemoryInactiveAssemblyValue : std::uint8_t {
  NotApplicable,
  ZeroBits,
};

class MemoryResultAssembly {
public:
  dataflow::semantics::ServiceValueRole role() const { return role_; }
  MemoryResultAssemblyStrategy strategy() const { return strategy_; }
  std::optional<std::uint64_t> laneCount() const {
    if (strategy_ != MemoryResultAssemblyStrategy::RowMajorLaneValues)
      return std::nullopt;
    return laneCount_;
  }
  std::optional<MemoryInactiveAssemblyValue> inactiveValue() const {
    if (strategy_ != MemoryResultAssemblyStrategy::RowMajorLaneValues)
      return std::nullopt;
    return inactiveValue_;
  }

private:
  MemoryResultAssembly(dataflow::semantics::ServiceValueRole role,
                       MemoryResultAssemblyStrategy strategy,
                       std::uint64_t laneCount,
                       MemoryInactiveAssemblyValue inactiveValue)
      : role_(role), strategy_(strategy), laneCount_(laneCount),
        inactiveValue_(inactiveValue) {}

  dataflow::semantics::ServiceValueRole role_;
  MemoryResultAssemblyStrategy strategy_;
  std::uint64_t laneCount_;
  MemoryInactiveAssemblyValue inactiveValue_;

  friend class MemoryPortAssembly;
};

enum class MemoryParentRetirement : std::uint8_t {
  SingleParentRetirement,
};

class MemoryPortAssembly {
public:
  llvm::ArrayRef<MemoryResultAssembly> results() const { return results_; }
  MemoryParentRetirement retirement() const { return retirement_; }

private:
  static MemoryPortAssembly
  derive(const dataflow::semantics::CanonicalService &parentService,
         MemoryPortTransactionProjection projection,
         const dataflow::semantics::CanonicalMemoryAccessView *access);

  explicit MemoryPortAssembly(std::vector<MemoryResultAssembly> results)
      : results_(std::move(results)),
        retirement_(MemoryParentRetirement::SingleParentRetirement) {}

  std::vector<MemoryResultAssembly> results_;
  MemoryParentRetirement retirement_;

  friend llvm::Expected<class MemoryPortTransactionPlan>
  deriveMemoryPortTransactionPlan(
      const MemoryOperationPatternView &pattern,
      const dataflow::CanonicalActorSchemaProjection &actor,
      const dataflow::semantics::CanonicalService &parentService,
      const std::optional<dataflow::semantics::CanonicalMemoryAccessView>
          &access);
};

class MemoryPortTransactionPlan {
public:
  const dataflow::semantics::CanonicalService &parentService() const {
    return parentService_;
  }
  llvm::ArrayRef<MemoryPortChildTransaction> transactions() const {
    return transactions_;
  }
  const MemoryPortAssembly &assembly() const { return assembly_; }

private:
  MemoryPortTransactionPlan(
      dataflow::semantics::CanonicalService parentService,
      std::vector<MemoryPortChildTransaction> transactions,
      MemoryPortAssembly assembly)
      : parentService_(std::move(parentService)),
        transactions_(std::move(transactions)), assembly_(std::move(assembly)) {
  }

  dataflow::semantics::CanonicalService parentService_;
  std::vector<MemoryPortChildTransaction> transactions_;
  MemoryPortAssembly assembly_;

  friend llvm::Expected<MemoryPortTransactionPlan>
  deriveMemoryPortTransactionPlan(
      const MemoryOperationPatternView &pattern,
      const dataflow::CanonicalActorSchemaProjection &actor,
      const dataflow::semantics::CanonicalService &parentService,
      const std::optional<dataflow::semantics::CanonicalMemoryAccessView>
          &access);
};

llvm::Expected<MemoryPortTransactionPlan> deriveMemoryPortTransactionPlan(
    const MemoryOperationPatternView &pattern,
    const dataflow::CanonicalActorSchemaProjection &actor,
    const dataflow::semantics::CanonicalService &parentService,
    const std::optional<dataflow::semantics::CanonicalMemoryAccessView>
        &access);

} // namespace fabric

#endif // LOOM_FABRIC_IR_MEMORY_PORT_TRANSACTION_H
