#include "Simulator/CgraClosedWaitCertificate.h"

#include "SimulationWireInternal.h"

#include "Common/ArtifactLocalReference.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::sim;

namespace {

using Diagnostic = CgraClosedWaitSetDiagnostic;
using loom::sim::detail::WireReader;
using loom::sim::detail::WireWriter;

constexpr std::uint64_t absent64 = std::numeric_limits<std::uint64_t>::max();

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::errc::invalid_argument,
      "cgra_closed_wait_certificate_invalid: %s", message.str().c_str());
}

void writeFramed(WireWriter &writer, llvm::ArrayRef<std::uint8_t> bytes) {
  writer.u64(bytes.size());
  writer.bytes(bytes);
}

llvm::Expected<llvm::ArrayRef<std::uint8_t>> readFramed(WireReader &reader) {
  auto size = reader.u64();
  if (!size)
    return size.takeError();
  if (*size > std::numeric_limits<std::size_t>::max())
    return invalid("framed byte count exceeds size_t");
  return reader.bytes(static_cast<std::size_t>(*size));
}

void writeRoot(WireWriter &writer, const ArtifactRootReference &reference) {
  writeFramed(writer, encodeArtifactRootReference(reference));
}

llvm::Expected<ArtifactRootReference> readRoot(WireReader &reader) {
  auto bytes = readFramed(reader);
  if (!bytes)
    return bytes.takeError();
  auto decoded = decodeArtifactRootReferencePrefix(*bytes);
  if (!decoded)
    return decoded.takeError();
  if (decoded->byteCount != bytes->size())
    return invalid("root reference has trailing bytes");
  return decoded->reference;
}

void writeApInt(WireWriter &writer, const llvm::APInt &value) {
  writer.u32(value.getBitWidth());
  const std::uint32_t byteCount = (value.getBitWidth() + 7) / 8;
  writer.u32(byteCount);
  const llvm::APInt extended = value.zextOrTrunc(byteCount * 8);
  std::vector<std::uint8_t> bytes;
  bytes.reserve(byteCount);
  for (std::uint32_t byte = 0; byte < byteCount; ++byte)
    bytes.push_back(static_cast<std::uint8_t>(extended.extractBitsAsZExtValue(
        8, 8 * (byteCount - 1 - byte))));
  writer.bytes(bytes);
}

llvm::Expected<llvm::APInt> readApInt(WireReader &reader) {
  auto bitWidth = reader.u32();
  if (!bitWidth)
    return bitWidth.takeError();
  auto byteCount = reader.u32();
  if (!byteCount)
    return byteCount.takeError();
  if (*bitWidth == 0 || *byteCount != (*bitWidth + 7) / 8)
    return invalid("APInt width and byte count disagree");
  auto bytes = reader.bytes(*byteCount);
  if (!bytes)
    return bytes.takeError();
  llvm::APInt value(*byteCount * 8, 0);
  for (std::uint8_t octet : *bytes)
    value = value.shl(8) | llvm::APInt(value.getBitWidth(), octet);
  return value.trunc(*bitWidth);
}

void writeOptionalApInt(WireWriter &writer,
                        const std::optional<llvm::APInt> &value) {
  writer.u32(value ? 1 : 0);
  if (value)
    writeApInt(writer, *value);
}

llvm::Expected<std::optional<llvm::APInt>>
readOptionalApInt(WireReader &reader) {
  auto present = reader.u32();
  if (!present)
    return present.takeError();
  if (*present > 1)
    return invalid("optional APInt presence is not boolean");
  if (*present == 0)
    return std::optional<llvm::APInt>();
  auto value = readApInt(reader);
  if (!value)
    return value.takeError();
  return std::optional<llvm::APInt>(std::move(*value));
}

void writeQueueClass(WireWriter &writer,
                     const Diagnostic::WaitQueueClass &queueClass) {
  writer.u32(queueClass.tagLocal ? 1 : 0);
  writeApInt(writer, queueClass.tagValue);
}

llvm::Expected<Diagnostic::WaitQueueClass>
readQueueClass(WireReader &reader) {
  auto tagLocal = reader.u32();
  if (!tagLocal)
    return tagLocal.takeError();
  if (*tagLocal > 1)
    return invalid("queue-class locality is not boolean");
  auto value = readApInt(reader);
  if (!value)
    return value.takeError();
  return Diagnostic::WaitQueueClass{*tagLocal != 0, std::move(*value)};
}

void writeOwner(WireWriter &writer, const Diagnostic::WaitOwnerKey &owner) {
  writer.u32(owner.owner.index());
  if (const auto *firing =
          std::get_if<Diagnostic::WaitActorFiringKey>(&owner.owner)) {
    writer.u64(firing->semanticActorOrdinal);
    writer.u64(firing->occurrenceOrdinal);
    return;
  }
  const auto &storage =
      std::get<Diagnostic::WaitStorageQueueKey>(owner.owner);
  writer.u32(static_cast<std::uint32_t>(storage.domain));
  writer.u64(storage.ordinal);
  writeQueueClass(writer, storage.queueClass);
}

llvm::Expected<Diagnostic::WaitOwnerKey> readOwner(WireReader &reader) {
  auto kind = reader.u32();
  if (!kind)
    return kind.takeError();
  if (*kind == 0) {
    auto actor = reader.u64();
    if (!actor)
      return actor.takeError();
    auto occurrence = reader.u64();
    if (!occurrence)
      return occurrence.takeError();
    return Diagnostic::WaitOwnerKey{
        Diagnostic::WaitActorFiringKey{*actor, *occurrence}};
  }
  if (*kind != 1)
    return invalid("wait-owner discriminant is unknown");
  auto domain = reader.u32();
  if (!domain)
    return domain.takeError();
  if (*domain >
      static_cast<std::uint32_t>(Diagnostic::WaitStorageDomain::OperandQueue))
    return invalid("wait-storage domain is unknown");
  auto ordinal = reader.u64();
  if (!ordinal)
    return ordinal.takeError();
  auto queueClass = readQueueClass(reader);
  if (!queueClass)
    return queueClass.takeError();
  return Diagnostic::WaitOwnerKey{Diagnostic::WaitStorageQueueKey{
      static_cast<Diagnostic::WaitStorageDomain>(*domain), *ordinal,
      std::move(*queueClass)}};
}

void writeOptionalFifo(
    WireWriter &writer,
    const std::optional<::loom::fabric::FabricFifoOccurrenceRef> &fifo) {
  writer.u32(fifo ? 1 : 0);
  if (fifo)
    writeFramed(writer, ::loom::fabric::canonicalFabricBytes(*fifo));
}

llvm::Expected<std::optional<::loom::fabric::FabricFifoOccurrenceRef>>
readOptionalFifo(WireReader &reader) {
  auto present = reader.u32();
  if (!present)
    return present.takeError();
  if (*present > 1)
    return invalid("optional FIFO presence is not boolean");
  if (*present == 0)
    return std::optional<::loom::fabric::FabricFifoOccurrenceRef>();
  auto bytes = readFramed(reader);
  if (!bytes)
    return bytes.takeError();
  auto fifo = ::loom::fabric::decodeFabricRef<
      ::loom::fabric::FabricFifoOccurrenceRef>(*bytes);
  if (!fifo)
    return fifo.takeError();
  return std::optional<::loom::fabric::FabricFifoOccurrenceRef>(*fifo);
}

std::vector<std::uint8_t> encodeEdge(const Diagnostic::WaitEdge &edge) {
  WireWriter writer;
  writeOwner(writer, edge.from);
  writeOwner(writer, edge.to);
  writer.u32(static_cast<std::uint32_t>(edge.kind));
  writer.u32(edge.waitingInputOrdinal);
  writer.u64(edge.waitingChannelOrdinal);
  writer.u64(edge.bindingOrdinal);
  writer.u64(edge.occurrenceOrdinal);
  writer.u64(edge.storageOrdinal);
  writeOptionalFifo(writer, edge.fifoOccurrence);
  writer.u32(edge.storageCapacity);
  writer.u32(edge.storageOccupancy);
  writer.u32(edge.awaitedClassPosition);
  writeOptionalApInt(writer, edge.awaitedTagValue);
  writeOptionalApInt(writer, edge.headTagValue);
  writer.u64(edge.headBindingOrdinal);
  writer.u64(edge.headOccurrenceOrdinal);
  writer.u64(edge.headDestinationActorOrdinal);
  writer.u32(edge.headDestinationInputOrdinal);
  writer.u64(edge.headDestinationChannelOrdinal);
  return writer.take();
}

llvm::Expected<Diagnostic::WaitEdge> decodeEdge(
    llvm::ArrayRef<std::uint8_t> bytes) {
  WireReader reader(bytes);
  Diagnostic::WaitEdge edge;
  auto from = readOwner(reader);
  if (!from)
    return from.takeError();
  edge.from = std::move(*from);
  auto to = readOwner(reader);
  if (!to)
    return to.takeError();
  edge.to = std::move(*to);
  auto kind = reader.u32();
  if (!kind)
    return kind.takeError();
  if (*kind > static_cast<std::uint32_t>(Diagnostic::WaitEdgeKind::OperandQueueWait))
    return invalid("wait-edge kind is unknown");
  edge.kind = static_cast<Diagnostic::WaitEdgeKind>(*kind);
  auto waitingInput = reader.u32();
  if (!waitingInput)
    return waitingInput.takeError();
  edge.waitingInputOrdinal = *waitingInput;
#define LOOM_READ_U64(Field)                                                   \
  do {                                                                         \
    auto value = reader.u64();                                                  \
    if (!value)                                                                \
      return value.takeError();                                                 \
    edge.Field = *value;                                                        \
  } while (false)
  LOOM_READ_U64(waitingChannelOrdinal);
  LOOM_READ_U64(bindingOrdinal);
  LOOM_READ_U64(occurrenceOrdinal);
  LOOM_READ_U64(storageOrdinal);
#undef LOOM_READ_U64
  auto fifo = readOptionalFifo(reader);
  if (!fifo)
    return fifo.takeError();
  edge.fifoOccurrence = std::move(*fifo);
  auto capacity = reader.u32();
  if (!capacity)
    return capacity.takeError();
  edge.storageCapacity = *capacity;
  auto occupancy = reader.u32();
  if (!occupancy)
    return occupancy.takeError();
  edge.storageOccupancy = *occupancy;
  auto position = reader.u32();
  if (!position)
    return position.takeError();
  edge.awaitedClassPosition = *position;
  auto awaitedTag = readOptionalApInt(reader);
  if (!awaitedTag)
    return awaitedTag.takeError();
  edge.awaitedTagValue = std::move(*awaitedTag);
  auto headTag = readOptionalApInt(reader);
  if (!headTag)
    return headTag.takeError();
  edge.headTagValue = std::move(*headTag);
#define LOOM_READ_U64(Field)                                                   \
  do {                                                                         \
    auto value = reader.u64();                                                  \
    if (!value)                                                                \
      return value.takeError();                                                 \
    edge.Field = *value;                                                        \
  } while (false)
  LOOM_READ_U64(headBindingOrdinal);
  LOOM_READ_U64(headOccurrenceOrdinal);
  LOOM_READ_U64(headDestinationActorOrdinal);
#undef LOOM_READ_U64
  auto headInput = reader.u32();
  if (!headInput)
    return headInput.takeError();
  edge.headDestinationInputOrdinal = *headInput;
  auto headChannel = reader.u64();
  if (!headChannel)
    return headChannel.takeError();
  edge.headDestinationChannelOrdinal = *headChannel;
  if (!reader.atEnd())
    return invalid("wait edge has trailing bytes");
  return edge;
}

void writeTraversalVector(
    WireWriter &writer,
    llvm::ArrayRef<::loom::fabric::FabricPhysicalTraversalRef> traversals) {
  writer.u64(traversals.size());
  for (const auto &traversal : traversals)
    writeFramed(writer, ::loom::fabric::canonicalFabricBytes(traversal));
}

llvm::Expected<std::vector<::loom::fabric::FabricPhysicalTraversalRef>>
readTraversalVector(WireReader &reader) {
  auto count = reader.u64();
  if (!count)
    return count.takeError();
  if (llvm::Error error = reader.guardCount(*count, 8))
    return std::move(error);
  std::vector<::loom::fabric::FabricPhysicalTraversalRef> result;
  result.reserve(static_cast<std::size_t>(*count));
  for (std::uint64_t index = 0; index < *count; ++index) {
    auto bytes = readFramed(reader);
    if (!bytes)
      return bytes.takeError();
    auto traversal = ::loom::fabric::decodeFabricRef<
        ::loom::fabric::FabricPhysicalTraversalRef>(*bytes);
    if (!traversal)
      return traversal.takeError();
    result.push_back(std::move(*traversal));
  }
  return result;
}

llvm::Expected<std::vector<std::uint8_t>> encodeTransfer(
    const CgraClosedWaitTransfer &transfer,
    const ArtifactIdentity &dataflowIdentity) {
  WireWriter writer;
  writer.u64(transfer.bindingOrdinal);
  writer.u64(transfer.occurrenceOrdinal);
  writeApInt(writer, transfer.physicalTagValue);
  writer.u32(transfer.tagged ? 1 : 0);
  auto encoded =
      ::dataflow::encodeDataflowReference(dataflowIdentity, transfer.producer);
  if (!encoded)
    return encoded.takeError();
  writeFramed(writer, *encoded);
  writer.u32(transfer.physicalTagOwner ? 1 : 0);
  if (transfer.physicalTagOwner) {
    writer.u32(transfer.physicalTagOwner->index());
    if (const auto *route =
            std::get_if<CgraRoutePhysicalTagOwner>(
                &*transfer.physicalTagOwner)) {
      auto owner = ::dataflow::encodeDataflowReference(dataflowIdentity,
                                                       route->producer);
      if (!owner)
        return owner.takeError();
      writeFramed(writer, *owner);
      writer.u64(route->segmentOrdinal);
    } else {
      const auto &local =
          std::get<CgraRegisterFifoPhysicalTagOwner>(
              *transfer.physicalTagOwner);
      auto owner = ::dataflow::encodeDataflowReference(dataflowIdentity,
                                                       local.producer);
      if (!owner)
        return owner.takeError();
      writeFramed(writer, *owner);
      auto consumer = ::dataflow::encodeDataflowReference(dataflowIdentity,
                                                           local.consumer);
      if (!consumer)
        return consumer.takeError();
      writeFramed(writer, *consumer);
    }
  }
  writer.u64(transfer.blockingStorageOrdinal);
  writer.u64(transfer.blockingDownstreamStorageOrdinal);
  writeTraversalVector(writer, transfer.blockingTraversals);
  writeTraversalVector(writer, transfer.blockingDownstreamTraversals);
  return writer.take();
}

llvm::Expected<CgraClosedWaitTransfer>
decodeTransfer(llvm::ArrayRef<std::uint8_t> bytes,
               const ArtifactIdentity &dataflowIdentity) {
  WireReader reader(bytes);
  auto binding = reader.u64();
  if (!binding)
    return binding.takeError();
  auto occurrence = reader.u64();
  if (!occurrence)
    return occurrence.takeError();
  auto tag = readApInt(reader);
  if (!tag)
    return tag.takeError();
  auto tagged = reader.u32();
  if (!tagged)
    return tagged.takeError();
  if (*tagged > 1)
    return invalid("transfer tagged state is not boolean");
  auto producerBytes = readFramed(reader);
  if (!producerBytes)
    return producerBytes.takeError();
  auto producer = ::dataflow::decodeDataflowReference<
      ::dataflow::CanonicalGraphProducerEndpointRef>(*producerBytes,
                                                      dataflowIdentity);
  if (!producer)
    return producer.takeError();
  CgraClosedWaitTransfer transfer(*binding, *occurrence, std::move(*tag),
                                  *tagged != 0, std::move(*producer));
  auto hasTagOwner = reader.u32();
  if (!hasTagOwner)
    return hasTagOwner.takeError();
  if (*hasTagOwner > 1)
    return invalid("Physical Tag owner presence is not boolean");
  if (*hasTagOwner) {
    auto kind = reader.u32();
    if (!kind)
      return kind.takeError();
    if (*kind > 1)
      return invalid("Physical Tag owner kind is unknown");
    auto ownerBytes = readFramed(reader);
    if (!ownerBytes)
      return ownerBytes.takeError();
    auto owner = ::dataflow::decodeDataflowReference<
        ::dataflow::CanonicalGraphProducerEndpointRef>(*ownerBytes,
                                                        dataflowIdentity);
    if (!owner)
      return owner.takeError();
    if (*kind == 0) {
      auto segment = reader.u64();
      if (!segment)
        return segment.takeError();
      transfer.physicalTagOwner =
          CgraRoutePhysicalTagOwner{std::move(*owner), *segment};
    } else {
      auto consumerBytes = readFramed(reader);
      if (!consumerBytes)
        return consumerBytes.takeError();
      auto consumer = ::dataflow::decodeDataflowReference<
          ::dataflow::CanonicalGraphConsumerEndpointRef>(*consumerBytes,
                                                          dataflowIdentity);
      if (!consumer)
        return consumer.takeError();
      transfer.physicalTagOwner = CgraRegisterFifoPhysicalTagOwner{
          std::move(*owner), std::move(*consumer)};
    }
  }
  auto storage = reader.u64();
  if (!storage)
    return storage.takeError();
  transfer.blockingStorageOrdinal = *storage;
  auto downstream = reader.u64();
  if (!downstream)
    return downstream.takeError();
  transfer.blockingDownstreamStorageOrdinal = *downstream;
  auto traversals = readTraversalVector(reader);
  if (!traversals)
    return traversals.takeError();
  transfer.blockingTraversals = std::move(*traversals);
  auto downstreamTraversals = readTraversalVector(reader);
  if (!downstreamTraversals)
    return downstreamTraversals.takeError();
  transfer.blockingDownstreamTraversals = std::move(*downstreamTraversals);
  if (!reader.atEnd())
    return invalid("transfer has trailing bytes");
  return transfer;
}

using TransferKey = std::pair<std::uint64_t, std::uint64_t>;

template <typename Transfer> TransferKey transferKey(const Transfer &transfer) {
  return {transfer.bindingOrdinal, transfer.occurrenceOrdinal};
}

} // namespace

llvm::Error loom::sim::verifyCgraClosedWaitCertificate(
    const CgraClosedWaitCertificate &certificate) {
  if (certificate.edges.empty())
    return invalid("certificate has no wait edges");
  if (!verifyClosedWaitCertificateClosure(certificate.edges))
    return invalid("wait edges are not one closed SCC");

  std::set<TransferKey> observed;
  std::optional<TransferKey> previousTransfer;
  for (const CgraClosedWaitTransfer &transfer : certificate.transfers) {
    const TransferKey key = transferKey(transfer);
    if (previousTransfer && !(*previousTransfer < key))
      return invalid("certificate transfers are not canonical");
    previousTransfer = key;
    if (!observed.insert(key).second)
      return invalid("certificate repeats a transfer");
    if (transfer.tagged != transfer.physicalTagOwner.has_value())
      return invalid("tagged transfer has no exact Mapping tag owner");
    if (!transfer.tagged &&
        transfer.physicalTagValue != llvm::APInt(1, 0))
      return invalid("untagged transfer has a noncanonical tag sentinel");
    if (transfer.physicalTagOwner) {
      const auto &ownerProducer = std::visit(
          [](const auto &owner)
              -> const ::dataflow::CanonicalGraphProducerEndpointRef & {
            return owner.producer;
          },
          *transfer.physicalTagOwner);
      if (ownerProducer != transfer.producer)
        return invalid("Physical Tag owner names a foreign logical producer");
    }
  }
  std::vector<std::uint8_t> previousEdge;
  for (const Diagnostic::WaitEdge &edge : certificate.edges) {
    std::vector<std::uint8_t> current = encodeEdge(edge);
    if (!previousEdge.empty() && !(previousEdge < current))
      return invalid("certificate edges are not canonical");
    previousEdge = std::move(current);
  }
  std::set<TransferKey> referenced;
  for (const Diagnostic::WaitEdge &edge : certificate.edges) {
    if (edge.bindingOrdinal != absent64)
      referenced.insert({edge.bindingOrdinal, edge.occurrenceOrdinal});
    if (edge.headBindingOrdinal != absent64)
      referenced.insert(
          {edge.headBindingOrdinal, edge.headOccurrenceOrdinal});
  }
  if (observed != referenced)
    return invalid("certificate transfer set differs from its edge references");
  return llvm::Error::success();
}

llvm::Expected<CgraClosedWaitCertificate>
loom::sim::buildCgraClosedWaitCertificate(
    const CgraClosedWaitSetDiagnostic &diagnostic) {
  if (!diagnostic.ownerReferences || diagnostic.waitProofFailure ||
      !verifyClosedWaitCertificateClosure(diagnostic))
    return invalid("diagnostic has no proven closed certificate");

  std::set<TransferKey> required;
  for (const Diagnostic::WaitEdge &edge : diagnostic.waitCertificate) {
    if (edge.bindingOrdinal != absent64)
      required.insert({edge.bindingOrdinal, edge.occurrenceOrdinal});
    if (edge.headBindingOrdinal != absent64)
      required.insert({edge.headBindingOrdinal, edge.headOccurrenceOrdinal});
  }
  std::map<TransferKey, const Diagnostic::Transfer *> available;
  for (const Diagnostic::Transfer &transfer : diagnostic.transfers)
    if (!available.try_emplace(transferKey(transfer), &transfer).second)
      return invalid("diagnostic repeats a transfer");

  CgraClosedWaitCertificate certificate(*diagnostic.ownerReferences);
  for (TransferKey key : required) {
    const auto found = available.find(key);
    if (found == available.end())
      return invalid("diagnostic omits a certificate transfer");
    const Diagnostic::Transfer &transfer = *found->second;
    if (!transfer.producer)
      return invalid("diagnostic certificate transfer has no logical producer");
    CgraClosedWaitTransfer persistent(
        transfer.bindingOrdinal, transfer.occurrenceOrdinal,
        transfer.physicalTagValue,
        transfer.physicalTagOrdinal != absent64, *transfer.producer);
    persistent.blockingStorageOrdinal = transfer.blockingStorageOrdinal;
    persistent.blockingDownstreamStorageOrdinal =
        transfer.blockingDownstreamStorageOrdinal;
    persistent.blockingTraversals = transfer.blockingTraversals;
    persistent.blockingDownstreamTraversals =
        transfer.blockingDownstreamTraversals;
    persistent.physicalTagOwner = transfer.physicalTagOwner;
    certificate.transfers.push_back(std::move(persistent));
  }
  certificate.edges = diagnostic.waitCertificate;
  llvm::sort(certificate.edges, [](const auto &lhs, const auto &rhs) {
    return encodeEdge(lhs) < encodeEdge(rhs);
  });
  if (llvm::Error error = verifyCgraClosedWaitCertificate(certificate))
    return std::move(error);
  return certificate;
}

llvm::Expected<std::vector<std::uint8_t>>
loom::sim::encodeCgraClosedWaitCertificate(
    const CgraClosedWaitCertificate &certificate) {
  if (llvm::Error error = verifyCgraClosedWaitCertificate(certificate))
    return std::move(error);
  WireWriter writer;
  writeRoot(writer, certificate.owners.dataflow);
  writeRoot(writer, certificate.owners.fabric);
  writeRoot(writer, certificate.owners.techMapping);
  writeRoot(writer, certificate.owners.spatialMapping);
  writer.u64(certificate.transfers.size());
  for (const CgraClosedWaitTransfer &transfer : certificate.transfers) {
    auto encoded =
        encodeTransfer(transfer, certificate.owners.dataflow.artifact);
    if (!encoded)
      return encoded.takeError();
    writeFramed(writer, *encoded);
  }
  writer.u64(certificate.edges.size());
  for (const Diagnostic::WaitEdge &edge : certificate.edges)
    writeFramed(writer, encodeEdge(edge));
  return writer.take();
}

llvm::Expected<CgraClosedWaitCertificate>
loom::sim::decodeCgraClosedWaitCertificate(llvm::ArrayRef<std::uint8_t> bytes) {
  WireReader reader(bytes);
  auto dataflow = readRoot(reader);
  if (!dataflow)
    return dataflow.takeError();
  auto fabric = readRoot(reader);
  if (!fabric)
    return fabric.takeError();
  auto tech = readRoot(reader);
  if (!tech)
    return tech.takeError();
  auto spatial = readRoot(reader);
  if (!spatial)
    return spatial.takeError();
  CgraClosedWaitCertificate certificate(CgraExecutionOwnerReferences{
      std::move(*dataflow), std::move(*fabric), std::move(*tech),
      std::move(*spatial)});

  auto transferCount = reader.u64();
  if (!transferCount)
    return transferCount.takeError();
  if (llvm::Error error = reader.guardCount(*transferCount, 8))
    return std::move(error);
  certificate.transfers.reserve(static_cast<std::size_t>(*transferCount));
  for (std::uint64_t index = 0; index < *transferCount; ++index) {
    auto transferBytes = readFramed(reader);
    if (!transferBytes)
      return transferBytes.takeError();
    auto transfer = decodeTransfer(*transferBytes,
                                   certificate.owners.dataflow.artifact);
    if (!transfer)
      return transfer.takeError();
    certificate.transfers.push_back(std::move(*transfer));
  }
  auto edgeCount = reader.u64();
  if (!edgeCount)
    return edgeCount.takeError();
  if (llvm::Error error = reader.guardCount(*edgeCount, 8))
    return std::move(error);
  certificate.edges.reserve(static_cast<std::size_t>(*edgeCount));
  for (std::uint64_t index = 0; index < *edgeCount; ++index) {
    auto edgeBytes = readFramed(reader);
    if (!edgeBytes)
      return edgeBytes.takeError();
    auto edge = decodeEdge(*edgeBytes);
    if (!edge)
      return edge.takeError();
    certificate.edges.push_back(std::move(*edge));
  }
  if (!reader.atEnd())
    return invalid("certificate has trailing bytes");
  if (llvm::Error error = verifyCgraClosedWaitCertificate(certificate))
    return std::move(error);
  return certificate;
}

llvm::Expected<CgraClosedWaitCertificateDigest>
loom::sim::digestCgraClosedWaitCertificate(
    const CgraClosedWaitCertificate &certificate) {
  auto bytes = encodeCgraClosedWaitCertificate(certificate);
  if (!bytes)
    return bytes.takeError();
  const llvm::StringRef domain = cgraClosedWaitCertificateDigestDomain;
  auto digest = computeComponentViewDigest(
      llvm::ArrayRef<std::uint8_t>(
          reinterpret_cast<const std::uint8_t *>(domain.data()), domain.size()),
      *bytes);
  if (!digest)
    return digest.takeError();
  return CgraClosedWaitCertificateDigest(std::move(*digest));
}

std::string loom::sim::formatCgraClosedWaitCertificateDigest(
    const CgraClosedWaitCertificateDigest &digest) {
  return formatComponentViewDigestHex(digest.value());
}
