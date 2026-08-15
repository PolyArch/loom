#include "Mapping/Artifact/MappingArtifact.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include <map>
#include <set>
#include <vector>

namespace loom::mapping {

llvm::Expected<TechMemoryInternalConnectionLegality>
deriveTechMemoryInternalConnectionLegality(
    llvm::ArrayRef<TechMemoryInternalEdgeView> internalEdges,
    const ArtifactIdentity &dataflowOwner) {
  using Key = std::vector<std::uint8_t>;

  std::map<Key, std::pair<Key, Key>> consumerSources;
  std::map<Key, std::set<Key>> connectionProducers;
  for (const TechMemoryInternalEdgeView &edge : internalEdges) {
    auto producer =
        ::dataflow::encodeDataflowReference(dataflowOwner, edge.producer);
    if (!producer)
      return producer.takeError();
    auto consumer =
        ::dataflow::encodeDataflowReference(dataflowOwner, edge.consumer);
    if (!consumer)
      return consumer.takeError();
    Key connection = ::loom::fabric::canonicalFabricBytes(edge.connection);

    const auto source = std::make_pair(*producer, connection);
    auto [consumerUse, inserted] =
        consumerSources.try_emplace(*consumer, source);
    if (!inserted && consumerUse->second != source)
      return TechMemoryInternalConnectionLegality::ConsumerHasMultipleSources;

    auto &producers = connectionProducers[std::move(connection)];
    producers.insert(std::move(*producer));
    if (producers.size() != 1)
      return TechMemoryInternalConnectionLegality::
          ConnectionHasMultipleProducers;
  }
  return TechMemoryInternalConnectionLegality::Admissible;
}

} // namespace loom::mapping
