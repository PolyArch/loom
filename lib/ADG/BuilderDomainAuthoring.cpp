#include "ADG/Builder.h"

#include "BuilderInternal.h"

#include "llvm/Support/Error.h"

namespace loom::adg {

using detail::activeState;
using detail::checkDomainHandleOwner;
using detail::invalid;

namespace {
using DomainMemberRole =
    ::fabric::ModuleDomainAuthoringRelation::InternalMemberRole;
} // namespace

ModuleDomainMemberHandle FuNode::domainMember() const {
  return ModuleDomainMemberHandle::internal(state_, rootOrdinal_, operation_,
                                            DomainMemberRole::FuNode, 0);
}

ModuleDomainMemberHandle FuBuilder::domainMember() const {
  return ModuleDomainMemberHandle::internal(state_, rootOrdinal_, operation_,
                                            DomainMemberRole::Occurrence, 0);
}

ModuleDomainMemberHandle PeBuilder::domainMember() const {
  return ModuleDomainMemberHandle::internal(state_, rootOrdinal_, operation_,
                                            DomainMemberRole::Occurrence, 0);
}

llvm::Expected<ModuleDomainMemberHandle>
PeBuilder::instructionContextMember(std::size_t ordinal) const {
  if (ordinal >= instructionContexts_)
    return invalid("PE instruction context ordinal is outside the resident "
                   "context inventory");
  return ModuleDomainMemberHandle::internal(state_, rootOrdinal_, operation_,
                                            DomainMemberRole::InstructionContext,
                                            ordinal);
}

llvm::Expected<ModuleDomainMemberHandle>
MemoryResult::operationPortMember(std::size_t ordinal) const {
  if (ordinal >= operationPorts_.size())
    return invalid("memory operation port ordinal is outside the engine "
                   "declaration");
  return operationPorts_[ordinal];
}

llvm::Expected<ModuleDomainSlotHandle> SpatialCoreBuilder::declareDomainSlot(
    loom::fabric::FabricClockResetKind kind) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (rootOrdinal_ >= (*state)->spatialRoots.size())
    return invalid("SpatialCore handle has an invalid owner ordinal");
  detail::SpatialRootState &root = (*state)->spatialRoots[rootOrdinal_];
  if (root.closed)
    return invalid("SpatialCore is already closed");
  auto ordinal = root.domainRelation.declareSlot(kind);
  if (!ordinal)
    return ordinal.takeError();
  return ModuleDomainSlotHandle(state_, rootOrdinal_, kind, *ordinal);
}

llvm::Expected<ModuleDomainMemberHandle>
SpatialCoreBuilder::inputDomainMember(std::size_t ordinal) const {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (rootOrdinal_ >= (*state)->spatialRoots.size())
    return invalid("SpatialCore handle has an invalid owner ordinal");
  detail::SpatialRootState &root = (*state)->spatialRoots[rootOrdinal_];
  if (ordinal >= root.operation.getFunctionType().getNumInputs())
    return invalid("boundary domain member is outside the Module signature");
  return ModuleDomainMemberHandle::boundary(
      state_, rootOrdinal_, loom::fabric::FabricPortDirection::Input, ordinal);
}

llvm::Expected<ModuleDomainMemberHandle>
SpatialCoreBuilder::outputDomainMember(std::size_t ordinal) const {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (rootOrdinal_ >= (*state)->spatialRoots.size())
    return invalid("SpatialCore handle has an invalid owner ordinal");
  detail::SpatialRootState &root = (*state)->spatialRoots[rootOrdinal_];
  if (ordinal >= root.operation.getFunctionType().getNumResults())
    return invalid("boundary domain member is outside the Module signature");
  return ModuleDomainMemberHandle::boundary(
      state_, rootOrdinal_, loom::fabric::FabricPortDirection::Output, ordinal);
}

llvm::Error SpatialCoreBuilder::assignDomainSlot(
    const ModuleDomainMemberHandle &member,
    const ModuleDomainSlotHandle &slot) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (rootOrdinal_ >= (*state)->spatialRoots.size())
    return invalid("SpatialCore handle has an invalid owner ordinal");
  detail::SpatialRootState &root = (*state)->spatialRoots[rootOrdinal_];
  if (root.closed)
    return invalid("SpatialCore is already closed");
  if (llvm::Error error =
          checkDomainHandleOwner(*state, rootOrdinal_, slot.state_,
                                 slot.rootOrdinal_, "domain slot"))
    return error;
  if (llvm::Error error =
          checkDomainHandleOwner(*state, rootOrdinal_, member.state_,
                                 member.rootOrdinal_, "domain member"))
    return error;
  if (!member.internal_)
    return root.domainRelation.assignBoundary(member.direction_,
                                              member.ordinal_, slot.kind_,
                                              slot.ordinal_);
  return root.domainRelation.assignInternal(member.owner_, member.role_,
                                            member.ordinal_, slot.kind_,
                                            slot.ordinal_);
}

} // namespace loom::adg
