#include "Mapping/Artifact/MappingArtifact.h"

#include "TechMappingCanonicalKeyInternal.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

namespace loom::mapping {
namespace {

using Bytes = std::vector<std::uint8_t>;

std::size_t framedSize(llvm::ArrayRef<std::uint8_t> value) {
  return sizeof(std::uint64_t) + value.size();
}

std::size_t orderedRecordsSize(llvm::ArrayRef<Bytes> records) {
  std::size_t size = sizeof(std::uint64_t);
  for (const Bytes &record : records)
    size += record.size();
  return size;
}

std::size_t framedRecordsSize(llvm::ArrayRef<Bytes> records) {
  std::size_t size = sizeof(std::uint64_t);
  for (const Bytes &record : records)
    size += framedSize(record);
  return size;
}

std::size_t
orderedRecordRefsSize(llvm::ArrayRef<llvm::ArrayRef<std::uint8_t>> records) {
  std::size_t size = sizeof(std::uint64_t);
  for (llvm::ArrayRef<std::uint8_t> record : records)
    size += record.size();
  return size;
}

void appendU32(Bytes &result, std::uint32_t value) {
  for (unsigned byte = 0; byte < 4; ++byte)
    result.push_back(static_cast<std::uint8_t>(value >> (8 * (3 - byte))));
}

void appendU64(Bytes &result, std::uint64_t value) {
  for (unsigned byte = 0; byte < 8; ++byte)
    result.push_back(static_cast<std::uint8_t>(value >> (8 * (7 - byte))));
}

void appendBytes(Bytes &result, llvm::ArrayRef<std::uint8_t> value) {
  appendU64(result, value.size());
  result.insert(result.end(), value.begin(), value.end());
}

Bytes bytes(mlir::DenseI8ArrayAttr attribute) {
  Bytes result;
  result.reserve(attribute.size());
  for (std::int8_t byte : attribute.asArrayRef())
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

template <typename Attr> Bytes attrBytes(Attr attribute) {
  return bytes(attribute.getRecord());
}

template <typename Ref>
void appendFabricRef(Bytes &result, const Ref &reference) {
  appendBytes(result, ::loom::fabric::canonicalFabricBytes(reference));
}

template <typename Ref>
llvm::Error appendDataflowRef(Bytes &result, const ArtifactIdentity &owner,
                              const Ref &reference) {
  auto encoded = ::dataflow::encodeDataflowReference(owner, reference);
  if (!encoded)
    return encoded.takeError();
  appendBytes(result, *encoded);
  return llvm::Error::success();
}

Bytes computeActorKey(llvm::ArrayRef<std::uint8_t> actor,
                      llvm::ArrayRef<std::uint8_t> operation,
                      llvm::ArrayRef<std::uint64_t> operands,
                      llvm::ArrayRef<std::uint64_t> results) {
  Bytes key;
  key.reserve(framedSize(actor) + framedSize(operation) +
              2 * sizeof(std::uint64_t) +
              sizeof(std::uint64_t) * (operands.size() + results.size()));
  appendBytes(key, actor);
  appendBytes(key, operation);
  appendU64(key, operands.size());
  for (std::uint64_t port : operands)
    appendU64(key, port);
  appendU64(key, results.size());
  for (std::uint64_t port : results)
    appendU64(key, port);
  return key;
}

Bytes computeBoundaryKey(llvm::ArrayRef<std::uint8_t> actor,
                         std::uint32_t direction, std::uint64_t ordinal,
                         llvm::ArrayRef<std::uint8_t> port) {
  Bytes key;
  key.reserve(framedSize(actor) + sizeof(std::uint32_t) +
              sizeof(std::uint64_t) + framedSize(port));
  appendBytes(key, actor);
  appendU32(key, direction);
  appendU64(key, ordinal);
  appendBytes(key, port);
  return key;
}

Bytes memoryActorKey(llvm::ArrayRef<std::uint8_t> actor,
                     llvm::ArrayRef<std::uint8_t> operationPort,
                     llvm::ArrayRef<std::uint8_t> capability,
                     llvm::ArrayRef<Bytes> operands,
                     llvm::ArrayRef<Bytes> results) {
  Bytes key;
  key.reserve(framedSize(actor) + framedSize(operationPort) +
              framedSize(capability) + framedRecordsSize(operands) +
              framedRecordsSize(results));
  appendBytes(key, actor);
  appendBytes(key, operationPort);
  appendBytes(key, capability);
  appendU64(key, operands.size());
  for (const Bytes &port : operands)
    appendBytes(key, port);
  appendU64(key, results.size());
  for (const Bytes &port : results)
    appendBytes(key, port);
  return key;
}

Bytes memoryTerminalKey(mlir::Attribute terminal) {
  Bytes key;
  if (auto producer =
          mlir::dyn_cast<::mapping::GraphProducerEndpointRefAttr>(terminal)) {
    Bytes record = attrBytes(producer);
    key.reserve(sizeof(std::uint32_t) + framedSize(record));
    appendU32(key, 0);
    appendBytes(key, record);
  } else {
    Bytes record = attrBytes(
        mlir::cast<::mapping::GraphConsumerEndpointRefAttr>(terminal));
    key.reserve(sizeof(std::uint32_t) + framedSize(record));
    appendU32(key, 1);
    appendBytes(key, record);
  }
  return key;
}

Bytes memoryBoundaryKey(llvm::ArrayRef<std::uint8_t> terminal,
                        llvm::ArrayRef<std::uint8_t> endpoint) {
  Bytes key;
  key.reserve(terminal.size() + framedSize(endpoint));
  key.insert(key.end(), terminal.begin(), terminal.end());
  appendBytes(key, endpoint);
  return key;
}

Bytes memoryInternalEdgeKey(llvm::ArrayRef<std::uint8_t> producer,
                            llvm::ArrayRef<std::uint8_t> consumer,
                            llvm::ArrayRef<std::uint8_t> connection) {
  Bytes key;
  key.reserve(framedSize(producer) + framedSize(consumer) +
              framedSize(connection));
  appendBytes(key, producer);
  appendBytes(key, consumer);
  appendBytes(key, connection);
  return key;
}

void appendOrderedRecords(Bytes &key, std::vector<Bytes> records) {
  llvm::sort(records);
  key.reserve(key.size() + orderedRecordsSize(records));
  appendU64(key, records.size());
  for (const Bytes &record : records)
    key.insert(key.end(), record.begin(), record.end());
}

void appendOrderedRecordRefs(
    Bytes &key, llvm::ArrayRef<llvm::ArrayRef<std::uint8_t>> records) {
  llvm::SmallVector<llvm::ArrayRef<std::uint8_t>, 16> ordered(records);
  llvm::sort(ordered, [](llvm::ArrayRef<std::uint8_t> lhs,
                         llvm::ArrayRef<std::uint8_t> rhs) {
    return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(),
                                        rhs.end());
  });
  key.reserve(key.size() + orderedRecordRefsSize(records));
  appendU64(key, ordered.size());
  for (llvm::ArrayRef<std::uint8_t> record : ordered)
    key.insert(key.end(), record.begin(), record.end());
}

struct ComputeKeyParts final {
  std::vector<Bytes> actors;
  std::vector<Bytes> boundaries;
};

struct MemoryKeyParts final {
  std::vector<Bytes> actors;
  std::vector<Bytes> boundaries;
  std::vector<Bytes> internalEdges;
};

Bytes computePayloadKey(llvm::ArrayRef<std::uint8_t> owner,
                        ComputeKeyParts parts) {
  Bytes key;
  key.reserve(framedSize(owner) + orderedRecordsSize(parts.actors) +
              orderedRecordsSize(parts.boundaries));
  appendBytes(key, owner);
  appendOrderedRecords(key, std::move(parts.actors));
  appendOrderedRecords(key, std::move(parts.boundaries));
  return key;
}

Bytes memoryPayloadKey(llvm::ArrayRef<std::uint8_t> owner,
                       MemoryKeyParts parts) {
  Bytes key;
  key.reserve(framedSize(owner) + orderedRecordsSize(parts.actors) +
              orderedRecordsSize(parts.boundaries) +
              orderedRecordsSize(parts.internalEdges));
  appendBytes(key, owner);
  appendOrderedRecords(key, std::move(parts.actors));
  appendOrderedRecords(key, std::move(parts.boundaries));
  appendOrderedRecords(key, std::move(parts.internalEdges));
  return key;
}

} // namespace

llvm::Expected<Bytes>
canonicalTechMatchActorKey(const TechComputeActorView &actor,
                           const ArtifactIdentity &dataflowOwner) {
  auto actorBytes =
      ::dataflow::encodeDataflowReference(dataflowOwner, actor.actor);
  if (!actorBytes)
    return actorBytes.takeError();
  return computeActorKey(
      *actorBytes, ::loom::fabric::canonicalFabricBytes(actor.fabricOperation),
      actor.operandPorts, actor.resultPorts);
}

llvm::Expected<Bytes>
canonicalTechMatchActorKey(const TechMemoryActorView &actor,
                           const ArtifactIdentity &dataflowOwner) {
  auto actorBytes =
      ::dataflow::encodeDataflowReference(dataflowOwner, actor.actor);
  if (!actorBytes)
    return actorBytes.takeError();
  std::vector<Bytes> operands;
  std::vector<Bytes> results;
  llvm::transform(actor.operandPorts, std::back_inserter(operands),
                  [](const auto &port) {
                    return ::loom::fabric::canonicalFabricBytes(port);
                  });
  llvm::transform(actor.resultPorts, std::back_inserter(results),
                  [](const auto &port) {
                    return ::loom::fabric::canonicalFabricBytes(port);
                  });
  return memoryActorKey(
      *actorBytes, ::loom::fabric::canonicalFabricBytes(actor.operationPort),
      ::loom::fabric::canonicalFabricBytes(actor.capability), operands,
      results);
}

llvm::Expected<Bytes>
canonicalTechMatchRowKey(const TechComputeRealizationView &realization,
                         const ArtifactIdentity &dataflowOwner) {
  ComputeKeyParts parts;
  for (const TechComputeActorView &actor : realization.actors) {
    auto key = canonicalTechMatchActorKey(actor, dataflowOwner);
    if (!key)
      return key.takeError();
    parts.actors.push_back(std::move(*key));
  }
  for (const TechComputeBoundaryView &boundary : realization.boundaries) {
    auto actorBytes =
        ::dataflow::encodeDataflowReference(dataflowOwner, boundary.actor);
    if (!actorBytes)
      return actorBytes.takeError();
    parts.boundaries.push_back(computeBoundaryKey(
        *actorBytes, static_cast<std::uint32_t>(boundary.direction),
        boundary.portOrdinal,
        ::loom::fabric::canonicalFabricBytes(boundary.fabricPort)));
  }
  Bytes key;
  Bytes payload = computePayloadKey(
      ::loom::fabric::canonicalFabricBytes(realization.capabilityTemplate),
      std::move(parts));
  key.reserve(sizeof(std::uint32_t) + payload.size());
  appendU32(key, 0);
  key.insert(key.end(), payload.begin(), payload.end());
  return key;
}

llvm::Expected<Bytes> detail::canonicalTechMemoryRowKeyFromActorKeys(
    ::loom::fabric::FabricMemoryEngineTemplateRef engine,
    llvm::ArrayRef<llvm::ArrayRef<std::uint8_t>> canonicalActorKeys,
    llvm::ArrayRef<TechMemoryGraphBoundaryView> graphBoundaries,
    llvm::ArrayRef<TechMemoryInternalEdgeView> internalEdges,
    const ArtifactIdentity &dataflowOwner) {
  MemoryKeyParts parts;
  for (const TechMemoryGraphBoundaryView &boundary : graphBoundaries) {
    Bytes terminal;
    if (const auto *producer =
            std::get_if<::dataflow::CanonicalGraphProducerEndpointRef>(
                &boundary.terminal)) {
      appendU32(terminal, 0);
      if (llvm::Error error =
              appendDataflowRef(terminal, dataflowOwner, *producer))
        return std::move(error);
    } else {
      appendU32(terminal, 1);
      if (llvm::Error error = appendDataflowRef(
              terminal, dataflowOwner,
              std::get<::dataflow::CanonicalGraphConsumerEndpointRef>(
                  boundary.terminal)))
        return std::move(error);
    }
    parts.boundaries.push_back(memoryBoundaryKey(
        terminal, ::loom::fabric::canonicalFabricBytes(boundary.endpoint)));
  }
  for (const TechMemoryInternalEdgeView &edge : internalEdges) {
    auto producer =
        ::dataflow::encodeDataflowReference(dataflowOwner, edge.producer);
    if (!producer)
      return producer.takeError();
    auto consumer =
        ::dataflow::encodeDataflowReference(dataflowOwner, edge.consumer);
    if (!consumer)
      return consumer.takeError();
    parts.internalEdges.push_back(memoryInternalEdgeKey(
        *producer, *consumer,
        ::loom::fabric::canonicalFabricBytes(edge.connection)));
  }

  Bytes key;
  key.reserve(sizeof(std::uint32_t) +
              framedSize(::loom::fabric::canonicalFabricBytes(engine)) +
              orderedRecordRefsSize(canonicalActorKeys) +
              orderedRecordsSize(parts.boundaries) +
              orderedRecordsSize(parts.internalEdges));
  appendU32(key, 1);
  appendBytes(key, ::loom::fabric::canonicalFabricBytes(engine));
  appendOrderedRecordRefs(key, canonicalActorKeys);
  appendOrderedRecords(key, std::move(parts.boundaries));
  appendOrderedRecords(key, std::move(parts.internalEdges));
  return key;
}

llvm::Expected<Bytes>
canonicalTechMatchRowKey(const TechMemoryRealizationView &realization,
                         const ArtifactIdentity &dataflowOwner) {
  std::vector<Bytes> actorKeys;
  actorKeys.reserve(realization.actors.size());
  for (const TechMemoryActorView &actor : realization.actors) {
    auto key = canonicalTechMatchActorKey(actor, dataflowOwner);
    if (!key)
      return key.takeError();
    actorKeys.push_back(std::move(*key));
  }
  llvm::SmallVector<llvm::ArrayRef<std::uint8_t>, 16> actorKeyRefs;
  actorKeyRefs.reserve(actorKeys.size());
  for (const Bytes &key : actorKeys)
    actorKeyRefs.push_back(key);
  return detail::canonicalTechMemoryRowKeyFromActorKeys(
      realization.engine, actorKeyRefs, realization.graphBoundaries,
      realization.internalEdges, dataflowOwner);
}

namespace detail {

std::vector<std::uint8_t> canonicalTechChildKey(mlir::Operation &operation) {
  Bytes key;
  if (auto actor = mlir::dyn_cast<::mapping::ComputeActorOp>(operation)) {
    Bytes child = computeActorKey(
        attrBytes(actor.getActor()), attrBytes(actor.getFabricOp()),
        llvm::to_vector(llvm::map_range(
            actor.getOperandPorts(),
            [](auto port) { return static_cast<std::uint64_t>(port); })),
        llvm::to_vector(llvm::map_range(actor.getResultPorts(), [](auto port) {
          return static_cast<std::uint64_t>(port);
        })));
    key.reserve(sizeof(std::uint32_t) + child.size());
    appendU32(key, 0);
    key.insert(key.end(), child.begin(), child.end());
    return key;
  }
  if (auto boundary = mlir::dyn_cast<::mapping::ComputeBoundaryOp>(operation)) {
    Bytes child = computeBoundaryKey(
        attrBytes(boundary.getActor()),
        static_cast<std::uint32_t>(boundary.getDirection()),
        boundary.getPortOrdinal(), attrBytes(boundary.getFuPort()));
    key.reserve(sizeof(std::uint32_t) + child.size());
    appendU32(key, 1);
    key.insert(key.end(), child.begin(), child.end());
    return key;
  }
  if (auto actor = mlir::dyn_cast<::mapping::MemoryActorOp>(operation)) {
    std::vector<Bytes> operands;
    std::vector<Bytes> results;
    llvm::transform(
        actor.getOperandPorts(), std::back_inserter(operands),
        [](mlir::Attribute port) {
          return attrBytes(
              mlir::cast<::mapping::FabricMemoryEngineTemplateEndpointRefAttr>(
                  port));
        });
    llvm::transform(
        actor.getResultPorts(), std::back_inserter(results),
        [](mlir::Attribute port) {
          return attrBytes(
              mlir::cast<::mapping::FabricMemoryEngineTemplateEndpointRefAttr>(
                  port));
        });
    Bytes child = memoryActorKey(
        attrBytes(actor.getActor()), attrBytes(actor.getOperationPort()),
        attrBytes(actor.getCapability()), operands, results);
    key.reserve(sizeof(std::uint32_t) + child.size());
    appendU32(key, 2);
    key.insert(key.end(), child.begin(), child.end());
    return key;
  }
  if (auto boundary =
          mlir::dyn_cast<::mapping::MemoryGraphBoundaryOp>(operation)) {
    Bytes child = memoryBoundaryKey(memoryTerminalKey(boundary.getTerminal()),
                                    attrBytes(boundary.getEndpoint()));
    key.reserve(sizeof(std::uint32_t) + child.size());
    appendU32(key, 3);
    key.insert(key.end(), child.begin(), child.end());
    return key;
  }
  auto edge = mlir::cast<::mapping::MemoryInternalEdgeOp>(operation);
  Bytes child = memoryInternalEdgeKey(attrBytes(edge.getProducer()),
                                      attrBytes(edge.getConsumer()),
                                      attrBytes(edge.getConnection()));
  key.reserve(sizeof(std::uint32_t) + child.size());
  appendU32(key, 4);
  key.insert(key.end(), child.begin(), child.end());
  return key;
}

Bytes canonicalTechRealizationPayloadKey(
    ::mapping::ComputeRealizationOp realization) {
  ComputeKeyParts parts;
  for (auto actor :
       realization.getBody().front().getOps<::mapping::ComputeActorOp>()) {
    parts.actors.push_back(computeActorKey(
        attrBytes(actor.getActor()), attrBytes(actor.getFabricOp()),
        llvm::to_vector(llvm::map_range(
            actor.getOperandPorts(),
            [](auto port) { return static_cast<std::uint64_t>(port); })),
        llvm::to_vector(llvm::map_range(actor.getResultPorts(), [](auto port) {
          return static_cast<std::uint64_t>(port);
        }))));
  }
  for (auto boundary :
       realization.getBody().front().getOps<::mapping::ComputeBoundaryOp>())
    parts.boundaries.push_back(computeBoundaryKey(
        attrBytes(boundary.getActor()),
        static_cast<std::uint32_t>(boundary.getDirection()),
        boundary.getPortOrdinal(), attrBytes(boundary.getFuPort())));
  return computePayloadKey(attrBytes(realization.getCapabilityTemplate()),
                           std::move(parts));
}

Bytes canonicalTechRealizationPayloadKey(
    ::mapping::MemoryRealizationOp realization) {
  MemoryKeyParts parts;
  for (auto actor :
       realization.getBody().front().getOps<::mapping::MemoryActorOp>()) {
    std::vector<Bytes> operands;
    std::vector<Bytes> results;
    llvm::transform(
        actor.getOperandPorts(), std::back_inserter(operands),
        [](mlir::Attribute port) {
          return attrBytes(
              mlir::cast<::mapping::FabricMemoryEngineTemplateEndpointRefAttr>(
                  port));
        });
    llvm::transform(
        actor.getResultPorts(), std::back_inserter(results),
        [](mlir::Attribute port) {
          return attrBytes(
              mlir::cast<::mapping::FabricMemoryEngineTemplateEndpointRefAttr>(
                  port));
        });
    parts.actors.push_back(memoryActorKey(
        attrBytes(actor.getActor()), attrBytes(actor.getOperationPort()),
        attrBytes(actor.getCapability()), operands, results));
  }
  for (auto boundary :
       realization.getBody().front().getOps<::mapping::MemoryGraphBoundaryOp>())
    parts.boundaries.push_back(
        memoryBoundaryKey(memoryTerminalKey(boundary.getTerminal()),
                          attrBytes(boundary.getEndpoint())));
  for (auto edge :
       realization.getBody().front().getOps<::mapping::MemoryInternalEdgeOp>())
    parts.internalEdges.push_back(memoryInternalEdgeKey(
        attrBytes(edge.getProducer()), attrBytes(edge.getConsumer()),
        attrBytes(edge.getConnection())));
  return memoryPayloadKey(attrBytes(realization.getEngine()), std::move(parts));
}

} // namespace detail
} // namespace loom::mapping
