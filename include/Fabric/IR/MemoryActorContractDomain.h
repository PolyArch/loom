#ifndef LOOM_FABRIC_IR_MEMORY_ACTOR_CONTRACT_DOMAIN_H
#define LOOM_FABRIC_IR_MEMORY_ACTOR_CONTRACT_DOMAIN_H

#include "Dataflow/IR/OperationSchema.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace fabric {

struct CompareExchangeOrderingPair {
  dataflow::AtomicOrdering success;
  dataflow::AtomicOrdering failure;

  friend bool operator==(CompareExchangeOrderingPair lhs,
                         CompareExchangeOrderingPair rhs) {
    return lhs.success == rhs.success && lhs.failure == rhs.failure;
  }
};

struct LoadStorePlainContractClause {
  std::vector<bool> volatileValues;
};

struct LoadStoreAtomicContractClause {
  std::vector<dataflow::AtomicOrdering> orderings;
  std::vector<dataflow::SyncScopeProjection> syncScopes;
  std::vector<std::optional<dataflow::VectorAtomicGranularity>>
      vectorGranularityValues;
  std::vector<bool> volatileValues;
};

struct AtomicRmwContractClause {
  std::vector<dataflow::AtomicRmwKind> rmwKinds;
  std::vector<dataflow::AtomicOrdering> orderings;
  std::vector<dataflow::SyncScopeProjection> syncScopes;
  std::vector<std::optional<dataflow::VectorAtomicGranularity>>
      vectorGranularityValues;
  std::vector<bool> volatileValues;
};

struct CompareExchangeContractClause {
  std::vector<CompareExchangeOrderingPair> orderingPairs;
  std::vector<dataflow::SyncScopeProjection> syncScopes;
  std::vector<std::optional<dataflow::VectorAtomicGranularity>>
      vectorGranularityValues;
  std::vector<bool> weakValues;
  std::vector<bool> volatileValues;
};

struct FenceContractClause {
  std::vector<dataflow::AtomicOrdering> orderings;
  std::vector<dataflow::SyncScopeProjection> syncScopes;
};

using MemoryActorContractClause =
    std::variant<LoadStorePlainContractClause, LoadStoreAtomicContractClause,
                 AtomicRmwContractClause, CompareExchangeContractClause,
                 FenceContractClause>;

class MemoryActorContractDomain {
public:
  static llvm::Expected<MemoryActorContractDomain>
  create(dataflow::OperationSchemaId actorSchema,
         llvm::ArrayRef<MemoryActorContractClause> clauses);

  static llvm::Expected<MemoryActorContractDomain>
  fromCanonical(dataflow::OperationSchemaId actorSchema,
                llvm::ArrayRef<MemoryActorContractClause> clauses);

  dataflow::OperationSchemaId actorSchema() const { return actorSchema_; }
  llvm::ArrayRef<MemoryActorContractClause> clauses() const { return clauses_; }

  bool contains(const dataflow::CanonicalActorSchemaProjection &actor) const;

private:
  MemoryActorContractDomain(dataflow::OperationSchemaId actorSchema,
                            std::vector<MemoryActorContractClause> clauses)
      : actorSchema_(actorSchema), clauses_(std::move(clauses)) {}

  dataflow::OperationSchemaId actorSchema_;
  std::vector<MemoryActorContractClause> clauses_;
};

llvm::Expected<std::vector<std::uint8_t>>
encodeMemoryActorContractDomain(const MemoryActorContractDomain &domain);

llvm::Expected<MemoryActorContractDomain>
decodeMemoryActorContractDomain(llvm::ArrayRef<std::uint8_t> bytes,
                                mlir::MLIRContext *context);

} // namespace fabric

#endif // LOOM_FABRIC_IR_MEMORY_ACTOR_CONTRACT_DOMAIN_H
