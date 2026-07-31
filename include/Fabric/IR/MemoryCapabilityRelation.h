#ifndef LOOM_FABRIC_IR_MEMORY_CAPABILITY_RELATION_H
#define LOOM_FABRIC_IR_MEMORY_CAPABILITY_RELATION_H

#include "Fabric/IR/MemoryActorContractDomain.h"
#include "Fabric/IR/MemoryCapabilityDomains.h"
#include "Fabric/IR/ReducedProductRelation.h"
#include "Fabric/IR/ResourceContract.h"

#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>

namespace fabric::detail {

inline constexpr bool memoryAccessClassRelationGrouping[] = {
    false, false, true, true, true, true, true, true, true, true};
inline constexpr std::size_t memoryAccessClassRelationFieldCount =
    sizeof(memoryAccessClassRelationGrouping) /
    sizeof(memoryAccessClassRelationGrouping[0]);

/// Internal common relation row used by memory operation ports and memory
/// services. `physicalFacts` is an exact owner codec payload; it is never a
/// persistent generic property bag or a semantic authority.
struct MemoryCapabilityRelationEntry {
  MemoryActorContractDomain actorContractDomain;
  std::optional<ParameterizedMemoryAccessDomain> accessDomain;
  std::vector<std::uint8_t> physicalFacts;
  std::vector<UsePatternKey> admissibleUsePatterns;
};

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

/// Normalizes the shared actor/access admission relation. Equal physical facts
/// merge use-pattern sets and complete duplicate rows are rejected. Semantic
/// owners may impose stronger overlap constraints on the normalized relation.
llvm::Expected<std::vector<MemoryCapabilityRelationEntry>>
normalizeMemoryCapabilityRelation(
    mlir::MLIRContext *context,
    llvm::ArrayRef<MemoryCapabilityRelationEntry> entries);

/// Exact symbolic containment for canonical memory-access domains.
llvm::Expected<bool>
memoryAccessDomainCovers(const ParameterizedMemoryAccessDomain &superset,
                         const ParameterizedMemoryAccessDomain &subset);

/// Exact semantic overlap, excluding physical facts and use-pattern choices.
llvm::Expected<bool> memoryCapabilityDomainsOverlap(
    const MemoryActorContractDomain &leftActors,
    const std::optional<ParameterizedMemoryAccessDomain> &leftAccesses,
    const MemoryActorContractDomain &rightActors,
    const std::optional<ParameterizedMemoryAccessDomain> &rightAccesses);

} // namespace fabric::detail

#endif // LOOM_FABRIC_IR_MEMORY_CAPABILITY_RELATION_H
