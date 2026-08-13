#include "Fabric/IR/SystemServiceContract.h"

#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <set>
#include <system_error>
#include <utility>

using namespace loom::fabric;

namespace {

llvm::Error malformed(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "system_service_contract_malformed: " + message);
}

class Writer {
public:
  void u32(std::uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
  }

  void u64(std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
  }

  void i64(std::int64_t value) {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    u64(bits);
  }

  void blob(llvm::ArrayRef<std::uint8_t> bytes) {
    u64(bytes.size());
    bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
  }

  std::vector<std::uint8_t> take() { return std::move(bytes_); }

private:
  std::vector<std::uint8_t> bytes_;
};

class Reader {
public:
  explicit Reader(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<std::uint32_t> u32(llvm::StringRef field) {
    if (bytes_.size() < 4)
      return malformed(field + " is a truncated uint32");
    std::uint32_t value = 0;
    for (unsigned index = 0; index < 4; ++index)
      value = (value << 8) | bytes_[index];
    bytes_ = bytes_.drop_front(4);
    return value;
  }

  llvm::Expected<std::uint64_t> u64(llvm::StringRef field) {
    if (bytes_.size() < 8)
      return malformed(field + " is a truncated uint64");
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index)
      value = (value << 8) | bytes_[index];
    bytes_ = bytes_.drop_front(8);
    return value;
  }

  llvm::Expected<std::int64_t> i64(llvm::StringRef field) {
    auto bits = u64(field);
    if (!bits)
      return bits.takeError();
    std::int64_t value = 0;
    static_assert(sizeof(value) == sizeof(*bits));
    std::memcpy(&value, &*bits, sizeof(value));
    return value;
  }

  llvm::Expected<std::uint32_t> tag(std::uint32_t count,
                                    llvm::StringRef field) {
    auto value = u32(field);
    if (!value)
      return value.takeError();
    if (*value >= count)
      return malformed(field + " has an unknown discriminant");
    return *value;
  }

  llvm::Expected<std::uint64_t> count(std::size_t minimumEntryBytes,
                                      llvm::StringRef field) {
    auto value = u64(field);
    if (!value)
      return value.takeError();
    if (minimumEntryBytes != 0 && *value > bytes_.size() / minimumEntryBytes)
      return malformed(field + " exceeds remaining framing");
    return *value;
  }

  llvm::Expected<llvm::ArrayRef<std::uint8_t>> blob(llvm::StringRef field) {
    auto size = u64((field + " length").str());
    if (!size)
      return size.takeError();
    if (*size > bytes_.size())
      return malformed(field + " is truncated");
    llvm::ArrayRef<std::uint8_t> result =
        bytes_.take_front(static_cast<std::size_t>(*size));
    bytes_ = bytes_.drop_front(static_cast<std::size_t>(*size));
    return result;
  }

  llvm::Error finish() const {
    if (!bytes_.empty())
      return malformed("record has trailing bytes");
    return llvm::Error::success();
  }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
};

template <typename Ref> void writeRef(Writer &writer, const Ref &reference) {
  writer.blob(canonicalFabricBytes(reference));
}

template <typename Ref>
llvm::Expected<Ref> readRef(Reader &reader, llvm::StringRef field) {
  auto bytes = reader.blob(field);
  if (!bytes)
    return bytes.takeError();
  return decodeFabricRef<Ref>(*bytes);
}

llvm::Expected<std::vector<std::uint8_t>>
encodeRate(const ServiceRateContractRecord &rate) {
  Writer writer;
  writeRef(writer, rate.rateClock());
  writer.u64(rate.operationsPerWindow());
  writer.u64(rate.windowTicks());
  writer.u64(rate.maxOutstanding());
  if (const auto *bounded =
          std::get_if<::fabric::BoundedCompletion>(&rate.progress())) {
    writer.u32(0);
    writeRef(writer, bounded->progressClock);
    writer.u64(bounded->maxIssueToRetireTicks);
  } else {
    writer.u32(1);
  }
  return writer.take();
}

llvm::Expected<ServiceRateContractRecord>
decodeRate(llvm::ArrayRef<std::uint8_t> bytes) {
  Reader reader(bytes);
  auto rateClock = readRef<ClockDomainRef>(reader, "rate clock");
  if (!rateClock)
    return rateClock.takeError();
  auto operations = reader.u64("operations per window");
  if (!operations)
    return operations.takeError();
  auto window = reader.u64("window ticks");
  if (!window)
    return window.takeError();
  auto outstanding = reader.u64("maximum outstanding");
  if (!outstanding)
    return outstanding.takeError();
  auto progressTag = reader.tag(2, "service progress");
  if (!progressTag)
    return progressTag.takeError();
  ServiceProgress progress;
  if (*progressTag == 0) {
    auto progressClock = readRef<ClockDomainRef>(reader, "progress clock");
    if (!progressClock)
      return progressClock.takeError();
    auto maximum = reader.u64("maximum issue-to-retire ticks");
    if (!maximum)
      return maximum.takeError();
    progress.emplace<::fabric::BoundedCompletion>(
        ::fabric::BoundedCompletion{*progressClock, *maximum});
  } else {
    progress.emplace<::fabric::FairEventual>();
  }
  if (llvm::Error error = reader.finish())
    return std::move(error);
  return ServiceRateContractRecord::create(*rateClock, *operations, *window,
                                           *outstanding, std::move(progress));
}

llvm::Expected<std::vector<std::uint8_t>>
encodeCapability(const CanonicalServiceCapabilityRecord &capability) {
  Writer writer;
  auto kind = dataflow::encodeServiceKind(capability.kind());
  if (!kind)
    return kind.takeError();
  writer.blob(kind->bytes());
  writer.u32(static_cast<std::uint32_t>(capability.role()));

  if (const auto *message =
          std::get_if<MessageTransferCapabilityDomain>(&capability.domain())) {
    writer.u32(0);
    writer.u64(message->payloadTypes().size());
    for (mlir::Type type : message->payloadTypes()) {
      auto encoded = dataflow::encodeCanonicalType(type);
      if (!encoded)
        return encoded.takeError();
      writer.blob(encoded->bytes());
    }
    writer.u32(message->fixedVectors().has_value() ? 1 : 0);
    if (message->fixedVectors()) {
      writer.u64(message->fixedVectors()->elementTypes().size());
      for (mlir::Type type : message->fixedVectors()->elementTypes()) {
        auto encoded = dataflow::encodeCanonicalType(type);
        if (!encoded)
          return encoded.takeError();
        writer.blob(encoded->bytes());
      }
      writer.u64(message->fixedVectors()->maximumPayloadBits());
      writer.u32(message->fixedVectors()->maximumRank());
    }
    writer.u64(message->pointerFormats().size());
    for (const ::fabric::PointerFormat &format :
         message->pointerFormats().formats()) {
      writer.u32(format.addressSpace);
      writer.u32(format.representationBits);
      writer.u32(format.addressBits);
      writer.u32(static_cast<std::uint32_t>(format.kind));
    }
  } else if (const auto *addressed =
                 std::get_if<AddressedMemoryCapabilityDomain>(
                     &capability.domain())) {
    writer.u32(1);
    auto actors =
        ::fabric::encodeMemoryActorContractDomain(addressed->actorContracts());
    if (!actors)
      return actors.takeError();
    writer.blob(*actors);
    auto accesses =
        ::fabric::encodeParameterizedMemoryAccessDomain(addressed->accesses());
    if (!accesses)
      return accesses.takeError();
    writer.blob(*accesses);
    auto addresses = ::fabric::encodeUnsignedDomain(addressed->addressBytes());
    if (!addresses)
      return addresses.takeError();
    writer.blob(*addresses);
    writer.u64(addressed->serviceBeatWidthBits());
    writer.u32(addressed->consistencyDomain().has_value() ? 1 : 0);
    if (addressed->consistencyDomain())
      writeRef(writer, *addressed->consistencyDomain());
  } else {
    const auto &fence = std::get<FenceCapabilityDomain>(capability.domain());
    writer.u32(2);
    auto actors =
        ::fabric::encodeMemoryActorContractDomain(fence.actorContracts());
    if (!actors)
      return actors.takeError();
    writer.blob(*actors);
    writeRef(writer, fence.consistencyDomain());
  }

  auto rate = encodeRate(capability.rate());
  if (!rate)
    return rate.takeError();
  writer.blob(*rate);
  return writer.take();
}

llvm::Expected<CanonicalServiceCapabilityRecord>
decodeCapability(llvm::ArrayRef<std::uint8_t> bytes,
                 mlir::MLIRContext *context) {
  Reader reader(bytes);
  auto kindBytes = reader.blob("service kind");
  if (!kindBytes)
    return kindBytes.takeError();
  auto kind = dataflow::decodeServiceKind(*kindBytes);
  if (!kind)
    return kind.takeError();
  auto roleTag = reader.tag(2, "service endpoint role");
  if (!roleTag)
    return roleTag.takeError();
  const auto role = static_cast<CanonicalServiceEndpointRole>(*roleTag);
  auto domainTag = reader.tag(3, "service capability domain");
  if (!domainTag)
    return domainTag.takeError();

  std::optional<CanonicalServiceCapabilityDomain> domain;
  if (*domainTag == 0) {
    auto count = reader.count(8, "message payload count");
    if (!count)
      return count.takeError();
    std::vector<mlir::Type> payloads;
    payloads.reserve(static_cast<std::size_t>(*count));
    for (std::uint64_t index = 0; index < *count; ++index) {
      auto typeBytes = reader.blob("message payload type");
      if (!typeBytes)
        return typeBytes.takeError();
      auto type = dataflow::decodeCanonicalType(*typeBytes, context);
      if (!type)
        return type.takeError();
      payloads.push_back(*type);
    }
    auto fixedVectorsPresent =
        reader.tag(2, "fixed-vector message domain presence");
    if (!fixedVectorsPresent)
      return fixedVectorsPresent.takeError();
    std::optional<FixedVectorMessagePayloadDomain> fixedVectors;
    if (*fixedVectorsPresent == 1) {
      auto elementCount = reader.count(8, "fixed-vector message element count");
      if (!elementCount)
        return elementCount.takeError();
      std::vector<mlir::Type> elements;
      elements.reserve(static_cast<std::size_t>(*elementCount));
      for (std::uint64_t index = 0; index < *elementCount; ++index) {
        auto typeBytes = reader.blob("fixed-vector message element type");
        if (!typeBytes)
          return typeBytes.takeError();
        auto type = dataflow::decodeCanonicalType(*typeBytes, context);
        if (!type)
          return type.takeError();
        elements.push_back(*type);
      }
      auto maximumPayloadBits =
          reader.u64("fixed-vector message maximum payload bits");
      if (!maximumPayloadBits)
        return maximumPayloadBits.takeError();
      auto maximumRank = reader.u32("fixed-vector message maximum rank");
      if (!maximumRank)
        return maximumRank.takeError();
      auto decoded = FixedVectorMessagePayloadDomain::fromCanonical(
          elements, *maximumPayloadBits, *maximumRank);
      if (!decoded)
        return decoded.takeError();
      fixedVectors = std::move(*decoded);
    }
    auto pointerFormatCount = reader.count(16, "message pointer-format count");
    if (!pointerFormatCount)
      return pointerFormatCount.takeError();
    ::fabric::PointerFormatRelation pointerFormats;
    std::optional<::fabric::PointerFormat> previousPointerFormat;
    for (std::uint64_t index = 0; index < *pointerFormatCount; ++index) {
      auto addressSpace = reader.u32("message pointer address space");
      if (!addressSpace)
        return addressSpace.takeError();
      auto representationBits =
          reader.u32("message pointer representation bits");
      if (!representationBits)
        return representationBits.takeError();
      auto addressBits = reader.u32("message pointer address bits");
      if (!addressBits)
        return addressBits.takeError();
      auto kind = reader.tag(
          static_cast<std::uint32_t>(::loom::PointerLayoutKind::ExternalState) +
              1,
          "message pointer layout kind");
      if (!kind)
        return kind.takeError();
      const ::fabric::PointerFormat format{
          *addressSpace, *representationBits, *addressBits,
          static_cast<::loom::PointerLayoutKind>(*kind)};
      if (previousPointerFormat && !(previousPointerFormat.value() < format))
        return malformed("message pointer-format domain is not sorted and "
                         "unique");
      if (!pointerFormats.insert(format))
        return malformed("message pointer-format domain contains an invalid "
                         "format");
      previousPointerFormat = format;
    }
    auto message = MessageTransferCapabilityDomain::fromCanonical(
        payloads, std::move(fixedVectors), std::move(pointerFormats));
    if (!message)
      return message.takeError();
    domain.emplace(std::move(*message));
  } else if (*domainTag == 1) {
    auto actorBytes = reader.blob("addressed actor domain");
    if (!actorBytes)
      return actorBytes.takeError();
    auto actors =
        ::fabric::decodeMemoryActorContractDomain(*actorBytes, context);
    if (!actors)
      return actors.takeError();
    auto accessBytes = reader.blob("addressed access domain");
    if (!accessBytes)
      return accessBytes.takeError();
    auto accesses =
        ::fabric::decodeParameterizedMemoryAccessDomain(*accessBytes);
    if (!accesses)
      return accesses.takeError();
    auto addressBytes = reader.blob("address range domain");
    if (!addressBytes)
      return addressBytes.takeError();
    auto addresses = ::fabric::decodeUnsignedDomain(*addressBytes);
    if (!addresses)
      return addresses.takeError();
    auto beatWidth = reader.u64("service beat width");
    if (!beatWidth)
      return beatWidth.takeError();
    auto consistencyPresent = reader.tag(2, "consistency domain presence");
    if (!consistencyPresent)
      return consistencyPresent.takeError();
    std::optional<MemoryConsistencyDomainRef> consistency;
    if (*consistencyPresent == 1) {
      auto reference =
          readRef<MemoryConsistencyDomainRef>(reader, "consistency domain");
      if (!reference)
        return reference.takeError();
      consistency = *reference;
    }
    auto addressed = AddressedMemoryCapabilityDomain::create(
        std::move(*actors), std::move(*accesses), std::move(*addresses),
        *beatWidth, std::move(consistency));
    if (!addressed)
      return addressed.takeError();
    domain.emplace(std::move(*addressed));
  } else {
    auto actorBytes = reader.blob("fence actor domain");
    if (!actorBytes)
      return actorBytes.takeError();
    auto actors =
        ::fabric::decodeMemoryActorContractDomain(*actorBytes, context);
    if (!actors)
      return actors.takeError();
    auto consistency =
        readRef<MemoryConsistencyDomainRef>(reader, "consistency domain");
    if (!consistency)
      return consistency.takeError();
    auto fence =
        FenceCapabilityDomain::create(std::move(*actors), *consistency);
    if (!fence)
      return fence.takeError();
    domain.emplace(std::move(*fence));
  }

  auto rateBytes = reader.blob("service rate contract");
  if (!rateBytes)
    return rateBytes.takeError();
  auto rate = decodeRate(*rateBytes);
  if (!rate)
    return rate.takeError();
  if (llvm::Error error = reader.finish())
    return std::move(error);
  auto capability = CanonicalServiceCapabilityRecord::create(
      *kind, role, std::move(*domain), std::move(*rate));
  if (!capability)
    return capability.takeError();
  auto canonical = encodeCapability(*capability);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*canonical) != bytes)
    return malformed("service capability is not canonical");
  return capability;
}

llvm::Expected<std::vector<std::uint8_t>>
encodeTransform(const SystemServiceTransformRecord &record) {
  Writer writer;
  writer.u64(record.inputs().size());
  for (const FabricMemoryEndpointRef &input : record.inputs())
    writeRef(writer, input);
  writer.u64(record.outputs().size());
  for (const FabricMemoryEndpointRef &output : record.outputs())
    writeRef(writer, output);

  if (const auto *offset =
          std::get_if<AddressOffsetTransform>(&record.contract())) {
    writer.u32(0);
    writer.u32(offset->addressWidth);
    writer.i64(offset->signedOffset);
  } else if (const auto *mask =
                 std::get_if<AddressMaskXorTransform>(&record.contract())) {
    writer.u32(1);
    writer.u32(mask->addressWidth);
    writer.u64(mask->andMask);
    writer.u64(mask->xorMask);
  } else if (const auto *interleave =
                 std::get_if<StaticInterleaveTransform>(&record.contract())) {
    writer.u32(2);
    writer.u64(interleave->granuleBytes);
    writer.u64(interleave->outputCount);
  } else {
    const auto &coherent = std::get<CoherentMemoryTransform>(record.contract());
    writer.u32(3);
    writeRef(writer, coherent.consistencyDomain);
    writer.u64(coherent.regions.size());
    for (const auto &region : coherent.regions) {
      writeRef(writer, region.input);
      writeRef(writer, region.output);
    }
  }
  return writer.take();
}

llvm::Error validateCapabilityKeys(
    llvm::ArrayRef<CanonicalServiceCapabilityRecord> capabilities) {
  std::set<std::pair<std::uint32_t, std::uint32_t>> keys;
  const CanonicalServiceEndpointRole role = capabilities.front().role();
  const bool transport = capabilities.front().kind() ==
                         dataflow::semantics::ServiceKind::MessageTransfer;
  for (const CanonicalServiceCapabilityRecord &capability : capabilities) {
    if (capability.role() != role)
      return malformed("service endpoint mixes operation-relative roles");
    const bool capabilityTransport =
        capability.kind() == dataflow::semantics::ServiceKind::MessageTransfer;
    if (capabilityTransport != transport)
      return malformed("service endpoint mixes transport and memory planes");
    const auto key =
        std::make_pair(static_cast<std::uint32_t>(capability.kind()),
                       static_cast<std::uint32_t>(capability.role()));
    if (!keys.insert(key).second)
      return malformed("service capability set repeats one kind and role");
  }
  return llvm::Error::success();
}

} // namespace

CanonicalServiceEndpointPlane CanonicalServiceCapabilitySet::plane() const {
  return capabilities_.front().kind() ==
                 dataflow::semantics::ServiceKind::MessageTransfer
             ? CanonicalServiceEndpointPlane::Transport
             : CanonicalServiceEndpointPlane::Memory;
}

std::vector<std::uint8_t> loom::fabric::encodeSystemServiceEndpointOwnerRef(
    const SystemServiceEndpointOwnerRef &reference) {
  return canonicalFabricBytes(reference.owner());
}

llvm::Expected<SystemServiceEndpointOwnerRef>
loom::fabric::decodeSystemServiceEndpointOwnerRef(
    llvm::ArrayRef<std::uint8_t> bytes) {
  auto owner = decodeFabricRef<FabricInventoryOwnerRef>(bytes);
  if (!owner)
    return owner.takeError();
  auto reference = SystemServiceEndpointOwnerRef::create(std::move(*owner));
  if (!reference)
    return reference.takeError();
  const std::vector<std::uint8_t> canonical =
      encodeSystemServiceEndpointOwnerRef(*reference);
  if (llvm::ArrayRef<std::uint8_t>(canonical) != bytes)
    return malformed("service endpoint owner is not canonical");
  return reference;
}

llvm::Expected<CanonicalServiceCapabilitySet>
CanonicalServiceCapabilitySet::create(
    std::vector<CanonicalServiceCapabilityRecord> capabilities) {
  if (capabilities.empty())
    return malformed("service capability set must not be empty");
  std::vector<
      std::pair<std::vector<std::uint8_t>, CanonicalServiceCapabilityRecord>>
      ordered;
  ordered.reserve(capabilities.size());
  for (CanonicalServiceCapabilityRecord &capability : capabilities) {
    auto bytes = encodeCapability(capability);
    if (!bytes)
      return bytes.takeError();
    ordered.emplace_back(std::move(*bytes), std::move(capability));
  }
  llvm::sort(ordered, [](const auto &left, const auto &right) {
    return left.first < right.first;
  });
  std::vector<CanonicalServiceCapabilityRecord> normalized;
  normalized.reserve(ordered.size());
  std::vector<std::uint8_t> previous;
  bool hasPrevious = false;
  for (auto &[bytes, capability] : ordered) {
    if (!hasPrevious || previous != bytes)
      normalized.push_back(std::move(capability));
    previous = std::move(bytes);
    hasPrevious = true;
  }
  if (llvm::Error error = validateCapabilityKeys(normalized))
    return std::move(error);
  return CanonicalServiceCapabilitySet(std::move(normalized));
}

llvm::Expected<CanonicalServiceCapabilitySet>
CanonicalServiceCapabilitySet::fromCanonical(
    std::vector<CanonicalServiceCapabilityRecord> capabilities) {
  if (capabilities.empty())
    return malformed("service capability set must not be empty");
  std::vector<std::uint8_t> previous;
  bool hasPrevious = false;
  for (const CanonicalServiceCapabilityRecord &capability : capabilities) {
    auto bytes = encodeCapability(capability);
    if (!bytes)
      return bytes.takeError();
    if (hasPrevious && previous >= *bytes)
      return malformed("service capability set is not sorted and unique");
    previous = std::move(*bytes);
    hasPrevious = true;
  }
  if (llvm::Error error = validateCapabilityKeys(capabilities))
    return std::move(error);
  return CanonicalServiceCapabilitySet(std::move(capabilities));
}

llvm::Expected<std::vector<std::uint8_t>>
loom::fabric::encodeCanonicalServiceCapabilitySet(
    const CanonicalServiceCapabilitySet &capabilities) {
  Writer writer;
  writer.u64(capabilities.capabilities().size());
  for (const CanonicalServiceCapabilityRecord &capability :
       capabilities.capabilities()) {
    auto bytes = encodeCapability(capability);
    if (!bytes)
      return bytes.takeError();
    writer.blob(*bytes);
  }
  return writer.take();
}

llvm::Expected<CanonicalServiceCapabilitySet>
loom::fabric::decodeCanonicalServiceCapabilitySet(
    llvm::ArrayRef<std::uint8_t> bytes, mlir::MLIRContext *context) {
  if (!context)
    return malformed("service capability decode requires an MLIR context");
  Reader reader(bytes);
  auto count = reader.count(8, "service capability count");
  if (!count)
    return count.takeError();
  std::vector<CanonicalServiceCapabilityRecord> capabilities;
  capabilities.reserve(static_cast<std::size_t>(*count));
  for (std::uint64_t index = 0; index < *count; ++index) {
    auto capabilityBytes = reader.blob("service capability");
    if (!capabilityBytes)
      return capabilityBytes.takeError();
    auto capability = decodeCapability(*capabilityBytes, context);
    if (!capability)
      return capability.takeError();
    capabilities.push_back(std::move(*capability));
  }
  if (llvm::Error error = reader.finish())
    return std::move(error);
  auto result =
      CanonicalServiceCapabilitySet::fromCanonical(std::move(capabilities));
  if (!result)
    return result.takeError();
  auto canonical = encodeCanonicalServiceCapabilitySet(*result);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*canonical) != bytes)
    return malformed("service capability set is not canonical");
  return result;
}

llvm::Expected<std::vector<std::uint8_t>>
loom::fabric::encodeServiceLegCarrierAttachmentRecord(
    const ServiceLegCarrierAttachmentRecord &record) {
  Writer writer;
  writeRef(writer, record.endpoint());
  auto kind = dataflow::encodeServiceKind(record.kind());
  if (!kind)
    return kind.takeError();
  writer.blob(kind->bytes());
  writer.u64(record.legOrdinal());
  writer.u64(record.carriers().size());
  for (const FabricTransportEndpointRef &carrier : record.carriers())
    writeRef(writer, carrier);
  return writer.take();
}

llvm::Expected<ServiceLegCarrierAttachmentRecord>
loom::fabric::decodeServiceLegCarrierAttachmentRecord(
    llvm::ArrayRef<std::uint8_t> bytes) {
  Reader reader(bytes);
  auto endpoint =
      readRef<FabricMemoryEndpointRef>(reader, "service memory endpoint");
  if (!endpoint)
    return endpoint.takeError();
  auto kindBytes = reader.blob("service kind");
  if (!kindBytes)
    return kindBytes.takeError();
  auto kind = dataflow::decodeServiceKind(*kindBytes);
  if (!kind)
    return kind.takeError();
  auto legOrdinal = reader.u64("service leg ordinal");
  if (!legOrdinal)
    return legOrdinal.takeError();
  auto carrierCount = reader.count(8, "service leg carrier count");
  if (!carrierCount)
    return carrierCount.takeError();
  std::vector<FabricTransportEndpointRef> carriers;
  carriers.reserve(static_cast<std::size_t>(*carrierCount));
  for (std::uint64_t index = 0; index < *carrierCount; ++index) {
    auto carrier = readRef<FabricTransportEndpointRef>(
        reader, "service leg transport carrier");
    if (!carrier)
      return carrier.takeError();
    carriers.push_back(std::move(*carrier));
  }
  if (llvm::Error error = reader.finish())
    return std::move(error);
  auto record = ServiceLegCarrierAttachmentRecord::fromCanonical(
      std::move(*endpoint), *kind, *legOrdinal, std::move(carriers));
  if (!record)
    return record.takeError();
  auto canonical = encodeServiceLegCarrierAttachmentRecord(*record);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*canonical) != bytes)
    return malformed("service leg carrier attachment is not canonical");
  return record;
}

llvm::Expected<std::vector<std::uint8_t>>
loom::fabric::encodeSystemServiceTransformRecord(
    const SystemServiceTransformRecord &record) {
  return encodeTransform(record);
}

llvm::Expected<SystemServiceTransformRecord>
loom::fabric::decodeSystemServiceTransformRecord(
    llvm::ArrayRef<std::uint8_t> bytes) {
  Reader reader(bytes);
  auto inputCount = reader.count(8, "transform input count");
  if (!inputCount)
    return inputCount.takeError();
  std::vector<FabricMemoryEndpointRef> inputs;
  inputs.reserve(static_cast<std::size_t>(*inputCount));
  for (std::uint64_t index = 0; index < *inputCount; ++index) {
    auto input = readRef<FabricMemoryEndpointRef>(reader, "transform input");
    if (!input)
      return input.takeError();
    inputs.push_back(std::move(*input));
  }
  auto outputCount = reader.count(8, "transform output count");
  if (!outputCount)
    return outputCount.takeError();
  std::vector<FabricMemoryEndpointRef> outputs;
  outputs.reserve(static_cast<std::size_t>(*outputCount));
  for (std::uint64_t index = 0; index < *outputCount; ++index) {
    auto output = readRef<FabricMemoryEndpointRef>(reader, "transform output");
    if (!output)
      return output.takeError();
    outputs.push_back(std::move(*output));
  }
  auto tag = reader.tag(4, "service transform contract");
  if (!tag)
    return tag.takeError();
  std::optional<ServiceTransformContract> contract;
  if (*tag == 0) {
    auto width = reader.u32("AddressOffset width");
    if (!width)
      return width.takeError();
    auto offset = reader.i64("AddressOffset value");
    if (!offset)
      return offset.takeError();
    contract.emplace(AddressOffsetTransform{*width, *offset});
  } else if (*tag == 1) {
    auto width = reader.u32("AddressMaskXor width");
    if (!width)
      return width.takeError();
    auto andMask = reader.u64("AddressMaskXor and mask");
    if (!andMask)
      return andMask.takeError();
    auto xorMask = reader.u64("AddressMaskXor xor mask");
    if (!xorMask)
      return xorMask.takeError();
    contract.emplace(AddressMaskXorTransform{*width, *andMask, *xorMask});
  } else if (*tag == 2) {
    auto granule = reader.u64("StaticInterleave granule");
    if (!granule)
      return granule.takeError();
    auto count = reader.u64("StaticInterleave output count");
    if (!count)
      return count.takeError();
    contract.emplace(StaticInterleaveTransform{*granule, *count});
  } else {
    auto domain = readRef<MemoryConsistencyDomainRef>(
        reader, "CoherentMemory consistency domain");
    if (!domain)
      return domain.takeError();
    auto count = reader.count(16, "CoherentMemory region count");
    if (!count)
      return count.takeError();
    std::vector<CoherentMemoryRegionCorrespondence> regions;
    regions.reserve(static_cast<std::size_t>(*count));
    for (std::uint64_t index = 0; index < *count; ++index) {
      auto input = readRef<FabricMemoryServiceRegionRef>(
          reader, "CoherentMemory input region");
      if (!input)
        return input.takeError();
      auto output = readRef<FabricMemoryServiceRegionRef>(
          reader, "CoherentMemory output region");
      if (!output)
        return output.takeError();
      regions.push_back({std::move(*input), std::move(*output)});
    }
    contract.emplace(CoherentMemoryTransform{*domain, std::move(regions)});
  }
  if (llvm::Error error = reader.finish())
    return std::move(error);
  auto record = SystemServiceTransformRecord::create(
      std::move(inputs), std::move(outputs), std::move(*contract));
  if (!record)
    return record.takeError();
  auto canonical = encodeTransform(*record);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*canonical) != bytes)
    return malformed("service transform record is not canonical");
  return record;
}
