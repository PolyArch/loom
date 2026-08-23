#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"

#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Dataflow/IR/OperationSchema.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <set>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::mapping {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "tech_ordered_input_projection_invalid: " + message);
}

llvm::Expected<bool>
isExternalInput(const TechComputeRealizationView &realization,
                const ::dataflow::ActorTokenOperandRef &consumer) {
  bool found = false;
  for (const TechComputeBoundaryView &boundary : realization.boundaries) {
    if (boundary.actor != consumer.actor ||
        boundary.direction != ::loom::fabric::FabricPortDirection::Input ||
        boundary.portOrdinal != consumer.ordinal)
      continue;
    if (found)
      return invalid("Tech realization repeats an actor input boundary");
    found = true;
  }
  return found;
}

void appendSized(std::vector<std::uint8_t> &destination,
                 llvm::ArrayRef<std::uint8_t> bytes) {
  for (int shift = 56; shift >= 0; shift -= 8)
    destination.push_back(static_cast<std::uint8_t>(bytes.size() >> shift));
  destination.insert(destination.end(), bytes.begin(), bytes.end());
}

} // namespace

llvm::Expected<std::vector<TechComputeOrderedInputGroupView>>
deriveTechComputeOrderedInputGroups(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping) {
  struct OrderedMember final {
    std::vector<std::uint8_t> producerKey;
    std::vector<std::uint8_t> consumerKey;
    TechComputeOrderedInputMemberView value;
  };

  std::vector<TechComputeOrderedInputGroupView> result;
  for (const TechComputeRealizationView &realization :
       techMapping.computeRealizations()) {
    for (const TechComputeActorView &selectedActor : realization.actors) {
      auto actor = dataflow.resolve(selectedActor.actor);
      if (!actor)
        return actor.takeError();
      auto schema =
          ::dataflow::projectRegisteredActorSchemaProjection(actor->op);
      if (!schema)
        return schema.takeError();
      auto cases = ::dataflow::semantics::projectActorHandshakeCases(
          schema->schema, actor->op->getNumOperands(),
          actor->op->getNumResults());
      if (!cases)
        return cases.takeError();

      std::set<std::vector<std::uint8_t>> seenGroups;
      for (const ::dataflow::semantics::ActorHandshakeCase &firing : *cases) {
        std::map<std::vector<std::uint8_t>, OrderedMember> byProducer;
        for (std::uint32_t input : firing.consumedInputs) {
          const ::dataflow::ActorTokenOperandRef consumer{selectedActor.actor,
                                                          input};
          auto external = isExternalInput(realization, consumer);
          if (!external)
            return external.takeError();
          if (!*external)
            continue;
          auto producer = dataflow.graphProducer(
              ::dataflow::CanonicalGraphConsumerEndpointRef{consumer});
          if (!producer)
            return producer.takeError();
          auto producerKey = ::dataflow::encodeDataflowReference(
              dataflow.identity(), *producer);
          if (!producerKey)
            return producerKey.takeError();
          auto consumerKey = ::dataflow::encodeDataflowReference(
              dataflow.identity(), consumer);
          if (!consumerKey)
            return consumerKey.takeError();
          OrderedMember member{
              *producerKey, *consumerKey, {consumer, *producer}};
          auto [known, inserted] =
              byProducer.try_emplace(member.producerKey, member);
          if (!inserted && member.consumerKey < known->second.consumerKey)
            known->second = std::move(member);
        }
        if (byProducer.size() < 2)
          continue;

        std::vector<std::uint8_t> groupKey;
        std::vector<TechComputeOrderedInputMemberView> members;
        members.reserve(byProducer.size());
        for (auto &[producerKey, member] : byProducer) {
          appendSized(groupKey, producerKey);
          appendSized(groupKey, member.consumerKey);
          members.push_back(std::move(member.value));
        }
        if (!seenGroups.insert(std::move(groupKey)).second)
          continue;
        result.push_back({realization.entityId, selectedActor.actor,
                          actor->graph, std::move(members)});
      }
    }
  }
  return result;
}

} // namespace loom::mapping
