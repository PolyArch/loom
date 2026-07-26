#ifndef LOOM_FABRIC_IR_MEMORY_CAPABILITY_RELATION_H
#define LOOM_FABRIC_IR_MEMORY_CAPABILITY_RELATION_H

#include "Fabric/IR/MemoryActorContractDomain.h"
#include "Fabric/IR/MemoryCapabilityDomains.h"
#include "Fabric/IR/ReducedProductRelation.h"

#include "llvm/Support/Error.h"

#include <cstdint>

namespace fabric::detail {

/// Internal projection of one actor-contract clause into the shared reduced
/// product representation. The tag is the stable Fabric clause tag; fields
/// retain their semantic-owner atom bytes.
struct MemoryActorClauseRelation {
  std::uint32_t tag;
  ReducedProductRow fields;
};

llvm::Expected<MemoryActorClauseRelation>
projectMemoryActorContractClause(const MemoryActorContractClause &clause);

llvm::Expected<MemoryActorContractClause>
importMemoryActorContractClause(const MemoryActorClauseRelation &relation,
                                mlir::MLIRContext *context);

llvm::Expected<ReducedProductRow>
projectMemoryAccessClass(const MemoryAccessClass &accessClass);

llvm::Expected<MemoryAccessClass>
importMemoryAccessClass(const ReducedProductRow &relation);

} // namespace fabric::detail

#endif // LOOM_FABRIC_IR_MEMORY_CAPABILITY_RELATION_H
