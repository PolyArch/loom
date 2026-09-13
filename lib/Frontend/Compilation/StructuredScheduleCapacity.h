#ifndef LOOM_FRONTEND_COMPILATION_STRUCTUREDSCHEDULECAPACITY_H
#define LOOM_FRONTEND_COMPILATION_STRUCTUREDSCHEDULECAPACITY_H

#include "Frontend/Compilation/FabricCapabilityIndex.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom::frontend::detail {

/// Removable address support: an address computation that the Graph memory
/// owner folds into the access positions it feeds, so the Fabric never
/// realizes it as an actor. The rule is transitive, because that owner
/// resolves a complete address chain back to its root: an address whose only
/// consumers are other removable address computations is removable with them.
/// It is also exact. An address that reaches any consumer other than another
/// address computation's base or a selected root-relative access's address
/// operand -- a non-access use, an escape through any other operand, or a
/// store that writes the address itself -- is arithmetic the Fabric must
/// realize, and so is an address whose accesses were not selected
/// root-relative. This one predicate answers both the enumeration capacity
/// projection below and the materialization admission gate, which is what
/// keeps those two from disagreeing about the same address.
bool isRemovableAddressSupport(mlir::Operation *operation);

/// The concrete Fabric occurrences that admit one structured actor, using the
/// admission projection its canonical kind requires.
llvm::Expected<std::uint64_t>
admittingStructuredActorResources(mlir::Operation *operation,
                                  const FabricCapabilityIndex &fabric);

/// The replication bound and the one actor group that produced it. The group
/// is what a reader needs to know when a whole factor family is retired: the
/// bound alone does not say which resource ran out.
struct AggregateReplicationBound final {
  std::uint64_t factor = 0;
  llvm::StringRef bindingActor;
  std::uint64_t bindingMultiplicity = 0;
  std::uint64_t bindingResources = 0;
};

/// Bounds how often the loop body may be replicated by dividing each actor
/// group's admitted concrete Fabric occurrences by its body multiplicity.
/// Removable address support is not a group. This projection proves nothing
/// about placement, routing, contention, or performance.
llvm::Expected<AggregateReplicationBound>
aggregateUnrollCapacity(mlir::scf::ForOp loop,
                        const FabricCapabilityIndex &fabric);

} // namespace loom::frontend::detail

#endif // LOOM_FRONTEND_COMPILATION_STRUCTUREDSCHEDULECAPACITY_H
