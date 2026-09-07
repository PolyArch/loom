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
  if (owner.stage == ActionStage::Storage) {
    auto target = projectPhysicalUseTarget(*plan_, event.actionOrdinal);
    if (!target)
      return target.takeError();
    const auto &use = std::get<PhysicalUseTarget>(*target).usePattern;
    return CgraPhysicalTraceBinding{
        FabricUsePhysicalActionOccurrenceRef{GraphInvocationOccurrenceRef{0},
                                             use, event.occurrenceOrdinal},
        std::move(*target)};
  }
  if (owner.transferSlot >= inFlight_.size() ||
      !inFlight_[owner.transferSlot].active)
    return invalid("CGRA trace transport action names an inactive token");
  const InFlight &inFlight = inFlight_[owner.transferSlot];
  const TransferBinding &binding = graph_.bindings[inFlight.bindingOrdinal];

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
    if (owner.stage != ActionStage::Traversal)
      return projectPhysicalUseTarget(*plan_, event.actionOrdinal);
    if (owner.traversalNodeOrdinal >= graph_.traversalNodes.size())
      return invalid("CGRA trace traversal action has no selected node");
    const TraversalNodeBinding &node =
        graph_.traversalNodes[owner.traversalNodeOrdinal];
    if (node.targetTraversalCount == 0 ||
        node.targetTraversalOffset > graph_.traversalTargets.size() ||
        node.targetTraversalCount >
            graph_.traversalTargets.size() - node.targetTraversalOffset)
      return invalid("CGRA trace traversal target slice is malformed");
    return projectPhysicalTransferTarget(
        *plan_, event.actionOrdinal,
        llvm::ArrayRef(graph_.traversalTargets)
            .slice(node.targetTraversalOffset, node.targetTraversalCount));
  };
  auto target = projectTarget();
  if (!target)
    return target.takeError();
  return CgraPhysicalTraceBinding{
      TokenPhysicalActionOccurrenceRef{std::move(token),
                                       owner.localActionOrdinal},
      std::move(*target)};
}

} // namespace loom::sim::detail
