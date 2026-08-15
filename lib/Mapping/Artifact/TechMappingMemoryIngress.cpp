#include "Mapping/Artifact/MappingArtifact.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <set>
#include <vector>

namespace loom::mapping {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "invalid Tech memory external ingress relation: " + message);
}

void appendU32Be(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

llvm::Error appendComponent(std::vector<std::uint8_t> &bytes,
                            llvm::ArrayRef<std::uint8_t> component) {
  if (component.size() > std::numeric_limits<std::uint32_t>::max())
    return invalid("canonical component size exceeds u32");
  appendU32Be(bytes, static_cast<std::uint32_t>(component.size()));
  bytes.insert(bytes.end(), component.begin(), component.end());
  return llvm::Error::success();
}

bool isSelectedInternalEdge(
    const TechMemoryRealizationView &realization,
    const ::dataflow::CanonicalGraphProducerEndpointRef &producer,
    const ::dataflow::CanonicalGraphConsumerEndpointRef &consumer) {
  return llvm::any_of(
      realization.internalEdges, [&](const TechMemoryInternalEdgeView &edge) {
        return edge.producer == producer && edge.consumer == consumer;
      });
}

} // namespace

llvm::Expected<std::vector<TechMemoryExternalIngressView>>
deriveTechMemoryExternalIngresses(
    const TechMemoryRealizationView &realization,
    const ::dataflow::CanonicalDataflowProgramView &dataflow) {
  std::vector<TechMemoryExternalIngressView> result;
  for (const TechMemoryActorView &actor : realization.actors) {
    auto resolved = dataflow.resolve(actor.actor);
    if (!resolved)
      return resolved.takeError();
    auto service =
        ::dataflow::semantics::CanonicalService::forActor(resolved->op);
    if (!service)
      return service.takeError();
    if (actor.operandPorts.size() != service->arguments().size())
      return invalid("actor operand correspondence is incomplete");
    for (auto [ordinal, endpoint] : llvm::enumerate(actor.operandPorts)) {
      auto operand = service->argumentValue(resolved->op, ordinal);
      if (!operand)
        return operand.takeError();
      const ::dataflow::CanonicalGraphConsumerEndpointRef consumer =
          ::dataflow::ActorTokenOperandRef{actor.actor,
                                           (*operand)->getOperandNumber()};
      auto producer = dataflow.graphProducer(consumer);
      if (!producer)
        return producer.takeError();
      if (isSelectedInternalEdge(realization, *producer, consumer))
        continue;
      result.push_back({endpoint, std::move(*producer)});
    }
  }
  return result;
}

llvm::Expected<std::vector<std::uint8_t>> canonicalTechMemoryExternalIngressKey(
    const TechMemoryExternalIngressView &ingress,
    const ArtifactIdentity &dataflowOwner) {
  std::vector<std::uint8_t> key;
  const auto endpoint = ::loom::fabric::canonicalFabricBytes(ingress.endpoint);
  auto producer =
      ::dataflow::encodeDataflowReference(dataflowOwner, ingress.producer);
  if (!producer)
    return producer.takeError();
  if (llvm::Error error = appendComponent(key, endpoint))
    return std::move(error);
  if (llvm::Error error = appendComponent(key, *producer))
    return std::move(error);
  return key;
}

llvm::Expected<bool> techMemoryExternalIngressesAreDistinct(
    const TechMemoryRealizationView &realization,
    const ::dataflow::CanonicalDataflowProgramView &dataflow) {
  auto ingresses = deriveTechMemoryExternalIngresses(realization, dataflow);
  if (!ingresses)
    return ingresses.takeError();
  std::set<std::vector<std::uint8_t>> keys;
  for (const TechMemoryExternalIngressView &ingress : *ingresses) {
    auto key =
        canonicalTechMemoryExternalIngressKey(ingress, dataflow.identity());
    if (!key)
      return key.takeError();
    if (!keys.insert(std::move(*key)).second)
      return false;
  }
  return true;
}

} // namespace loom::mapping
