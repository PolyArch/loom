#include "CgraTransportRuntime.h"

#include <system_error>
#include <type_traits>
#include <utility>

namespace loom::sim::detail {
namespace {

llvm::Error invalid(llvm::Twine message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument), message);
}

} // namespace

llvm::Expected<CgraPhysicalTraceBinding>
CgraTransportRuntime::physicalTraceBinding(
    const CgraPhysicalLifecycleEvent &event) const {
  auto indexed =
      actionOwners_.find({event.actionOrdinal, event.occurrenceOrdinal});
  if (indexed == actionOwners_.end())
    return invalid("CGRA trace transport action has no active owner");
  const ActionOwner &owner = indexed->second;
  if (owner.transferSlot >= inFlight_.size() ||
      !inFlight_[owner.transferSlot].active)
    return invalid("CGRA trace transport action names an inactive token");
  if (owner.secondaryTransferSlot != invalidCgraTransportOrdinal)
    return llvm::createStringError(
        std::errc::not_supported,
        "CGRA trace cannot assign one simultaneous storage action to two "
        "token occurrences");
  const InFlight &inFlight = inFlight_[owner.transferSlot];
  const TransferBinding &binding = bindings_[inFlight.bindingOrdinal];

  TokenOccurrenceRef token = std::visit(
      [&](const auto &producer) -> TokenOccurrenceRef {
        using Producer = std::decay_t<decltype(producer)>;
        if constexpr (std::is_same_v<Producer,
                                     ::dataflow::GraphIngressTokenRef>) {
          return GraphIngressTokenOccurrenceRef{
              GraphInvocationOccurrenceRef{0}, producer,
              inFlight.producerSequenceOrdinal};
        } else {
          return ActorResultTokenOccurrenceRef{
              ActorTransitionOccurrenceRef{GraphInvocationOccurrenceRef{0},
                                           producer.actor,
                                           inFlight.occurrenceOrdinal},
              producer.ordinal, inFlight.producerSequenceOrdinal};
        }
      },
      binding.producer);

  auto projectTarget = [&]() -> llvm::Expected<PhysicalActionTarget> {
    if (owner.stage != ActionStage::Traversal &&
        owner.stage != ActionStage::Storage)
      return projectPhysicalUseTarget(*plan_, event.actionOrdinal);
    if (owner.traversalNodeOrdinal >= traversalNodes_.size())
      return invalid("CGRA trace traversal action has no selected node");
    const TraversalNodeBinding &node =
        traversalNodes_[owner.traversalNodeOrdinal];
    if (node.targetTraversalCount == 0 ||
        node.targetTraversalOffset > traversalTargets_.size() ||
        node.targetTraversalCount >
            traversalTargets_.size() - node.targetTraversalOffset)
      return invalid("CGRA trace traversal target slice is malformed");
    return projectPhysicalTransferTarget(
        *plan_, event.actionOrdinal,
        llvm::ArrayRef(traversalTargets_)
            .slice(node.targetTraversalOffset, node.targetTraversalCount));
  };
  auto target = projectTarget();
  if (!target)
    return target.takeError();
  return CgraPhysicalTraceBinding{
      PhysicalActionOccurrenceRef{TokenPhysicalActionParent{std::move(token)},
                                  owner.localActionOrdinal},
      std::move(*target)};
}

} // namespace loom::sim::detail
