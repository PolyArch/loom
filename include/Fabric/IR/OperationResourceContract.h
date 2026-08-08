#ifndef FABRIC_IR_OPERATIONRESOURCECONTRACT_H
#define FABRIC_IR_OPERATIONRESOURCECONTRACT_H

#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Fabric/IR/ResourceContract.h"

namespace fabric {

/// Reusable exact contract for a stateless operation with one registered
/// elastic result slot. A use acquires the slot in cycle t and publishes and
/// releases it in cycle t + 1. This is an explicit hardware building block,
/// not a default for fabric.op.
const ResourceContract &oneCycleElasticOperationResourceContract();

/// Tests exact equality with the canonical one-cycle elastic operation
/// contract without requiring consumers to duplicate its physical semantics.
llvm::Expected<bool>
isOneCycleElasticOperationResourceContract(const ResourceContract &contract);

/// Exact initial contracts for the four loop-control implementation families.
/// Every use-pattern ordinal is the ordinal of the corresponding schema-owned
/// transition case. The ResourceContract owns only physical state, atomic use,
/// commit, and timing; the Dataflow descriptor remains the sole owner of
/// consumed heads, active results, and logical next-state semantics.
const ResourceContract &loopStreamOperationResourceContract();
const ResourceContract &loopCarryOperationResourceContract();
const ResourceContract &loopInvariantOperationResourceContract();
const ResourceContract &loopGateOperationResourceContract();

/// Resolves one schema-owned actor transition case to the exact physical use
/// pattern selected by a concrete fabric.op contract. A single-pattern
/// contract shares that pattern across every transition case; otherwise the
/// canonical case ordinal selects the pattern directly.
llvm::Expected<UsePatternKey>
resolveOperationUsePattern(const ResourceContract &contract,
                           std::uint32_t transitionCaseOrdinal);

UsePatternKey
loopControlUsePattern(::dataflow::semantics::StreamCase transition);
UsePatternKey
loopControlUsePattern(::dataflow::semantics::CarryCase transition);
UsePatternKey
loopControlUsePattern(::dataflow::semantics::InvariantCase transition);
UsePatternKey loopControlUsePattern(::dataflow::semantics::GateCase transition);

} // namespace fabric

#endif // FABRIC_IR_OPERATIONRESOURCECONTRACT_H
