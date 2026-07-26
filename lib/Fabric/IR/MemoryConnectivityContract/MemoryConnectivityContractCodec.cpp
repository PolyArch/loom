#include "Fabric/IR/MemoryConnectivityContract.h"

#include <limits>
#include <system_error>

namespace fabric {
namespace {

enum class DispatchTargetWireTag : std::uint32_t {
  LocalMemoryService = 0,
  ManagerEndpoint = 1,
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(std::errc::invalid_argument,
                                 "invalid MemoryConnectivityContractRecord: %s",
                                 message.str().c_str());
}

class Writer final {
public:
  void u32(std::uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
  }
  void u64(std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
  }
  llvm::Error count(std::size_t value, llvm::StringRef field) {
    if (value > std::numeric_limits<std::uint64_t>::max())
      return invalid(field + " count exceeds u64");
    u64(static_cast<std::uint64_t>(value));
    return llvm::Error::success();
  }
  std::vector<std::uint8_t> take() { return std::move(bytes_); }

private:
  std::vector<std::uint8_t> bytes_;
};

class Reader final {
public:
  explicit Reader(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<std::uint32_t> u32(llvm::StringRef field) {
    if (remaining() < 4)
      return invalid("truncated " + field);
    std::uint32_t value = 0;
    for (unsigned index = 0; index < 4; ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }
  llvm::Expected<std::uint64_t> u64(llvm::StringRef field) {
    if (remaining() < 8)
      return invalid("truncated " + field);
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }
  llvm::Error finish() const {
    return remaining() == 0 ? llvm::Error::success()
                            : invalid("record has trailing bytes");
  }
  std::size_t remaining() const { return bytes_.size() - offset_; }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

llvm::Error appendTarget(Writer &writer, const MemoryDispatchTarget &target) {
  if (std::holds_alternative<LocalMemoryDispatchTarget>(target)) {
    writer.u32(
        static_cast<std::uint32_t>(DispatchTargetWireTag::LocalMemoryService));
    return llvm::Error::success();
  }
  writer.u32(
      static_cast<std::uint32_t>(DispatchTargetWireTag::ManagerEndpoint));
  writer.u64(std::get<ManagerMemoryDispatchTarget>(target).endpointOrdinal);
  return llvm::Error::success();
}

llvm::Expected<MemoryDispatchTarget> readTarget(Reader &reader) {
  auto tag = reader.u32("dispatch target tag");
  if (!tag)
    return tag.takeError();
  switch (*tag) {
  case static_cast<std::uint32_t>(DispatchTargetWireTag::LocalMemoryService):
    return MemoryDispatchTarget(std::in_place_type<LocalMemoryDispatchTarget>);
  case static_cast<std::uint32_t>(DispatchTargetWireTag::ManagerEndpoint): {
    auto ordinal = reader.u64("manager endpoint ordinal");
    if (!ordinal)
      return ordinal.takeError();
    return MemoryDispatchTarget(std::in_place_type<ManagerMemoryDispatchTarget>,
                                ManagerMemoryDispatchTarget{*ordinal});
  }
  default:
    return invalid("unknown dispatch target tag");
  }
}

llvm::Error appendTargets(Writer &writer,
                          llvm::ArrayRef<MemoryDispatchTarget> targets) {
  if (llvm::Error error = writer.count(targets.size(), "dispatch targets"))
    return error;
  for (const MemoryDispatchTarget &target : targets)
    if (llvm::Error error = appendTarget(writer, target))
      return error;
  return llvm::Error::success();
}

llvm::Expected<std::vector<MemoryDispatchTarget>> readTargets(Reader &reader) {
  auto count = reader.u64("dispatch target count");
  if (!count)
    return count.takeError();
  if (*count == 0 || *count > reader.remaining() / 4)
    return invalid("dispatch target count exceeds its framing");
  std::vector<MemoryDispatchTarget> targets;
  targets.reserve(*count);
  for (std::uint64_t index = 0; index < *count; ++index) {
    auto target = readTarget(reader);
    if (!target)
      return target.takeError();
    targets.push_back(std::move(*target));
  }
  return targets;
}

llvm::Expected<MemoryConnectivityDeclaration> readDeclaration(Reader &reader) {
  MemoryConnectivityDeclaration declaration;
  auto portCount = reader.u64("operation port count");
  if (!portCount)
    return portCount.takeError();
  if (*portCount > reader.remaining() / 8)
    return invalid("operation port count exceeds its framing");
  declaration.operationPorts.reserve(*portCount);
  for (std::uint64_t port = 0; port < *portCount; ++port) {
    auto capabilityCount = reader.u64("capability target-domain count");
    if (!capabilityCount)
      return capabilityCount.takeError();
    if (*capabilityCount == 0 || *capabilityCount > reader.remaining() / 8)
      return invalid("capability target-domain count exceeds its framing");
    MemoryOperationPortDispatchDeclaration portDeclaration;
    portDeclaration.capabilityTargetDomains.reserve(*capabilityCount);
    for (std::uint64_t capability = 0; capability < *capabilityCount;
         ++capability) {
      auto targets = readTargets(reader);
      if (!targets)
        return targets.takeError();
      portDeclaration.capabilityTargetDomains.push_back(std::move(*targets));
    }
    declaration.operationPorts.push_back(std::move(portDeclaration));
  }

  auto subordinateCount = reader.u64("subordinate endpoint count");
  if (!subordinateCount)
    return subordinateCount.takeError();
  if (*subordinateCount > reader.remaining() / 24)
    return invalid("subordinate endpoint count exceeds its framing");
  declaration.subordinateEndpoints.reserve(*subordinateCount);
  for (std::uint64_t endpoint = 0; endpoint < *subordinateCount; ++endpoint) {
    auto capacity = reader.u64("subordinate provider capacity");
    auto fieldCount = reader.u64("subordinate match-field count");
    if (!capacity)
      return capacity.takeError();
    if (!fieldCount)
      return fieldCount.takeError();
    if (*fieldCount > reader.remaining() / 4)
      return invalid("subordinate match-field count exceeds its framing");
    std::vector<MemoryProviderMatchField> fields;
    fields.reserve(*fieldCount);
    for (std::uint64_t index = 0; index < *fieldCount; ++index) {
      auto tag = reader.u32("subordinate match-field tag");
      if (!tag ||
          *tag > static_cast<std::uint32_t>(MemoryProviderMatchField::Context))
        return tag ? invalid("unknown subordinate match-field tag")
                   : tag.takeError();
      fields.push_back(static_cast<MemoryProviderMatchField>(*tag));
    }
    auto transform = reader.u32("subordinate address transform");
    if (!transform ||
        *transform > static_cast<std::uint32_t>(
                         MemoryProviderAddressTransform::ConstantBaseOffset))
      return transform ? invalid("unknown subordinate address transform")
                       : transform.takeError();
    auto targets = readTargets(reader);
    if (!targets)
      return targets.takeError();
    declaration.subordinateEndpoints.push_back(
        {*capacity, std::move(fields),
         static_cast<MemoryProviderAddressTransform>(*transform),
         std::move(*targets)});
  }

  auto connectionCount = reader.u64("internal connection count");
  if (!connectionCount)
    return connectionCount.takeError();
  if (*connectionCount > reader.remaining() / 16)
    return invalid("internal connection count exceeds its framing");
  declaration.internalConnections.reserve(*connectionCount);
  for (std::uint64_t index = 0; index < *connectionCount; ++index) {
    auto source = reader.u64("internal source endpoint");
    auto sink = reader.u64("internal sink endpoint");
    if (!source)
      return source.takeError();
    if (!sink)
      return sink.takeError();
    declaration.internalConnections.push_back({*source, *sink});
  }
  return declaration;
}

} // namespace

llvm::Expected<std::vector<std::uint8_t>>
encodeMemoryConnectivityContractRecord(
    const MemoryConnectivityContractRecord &record) {
  Writer writer;
  if (llvm::Error error =
          writer.count(record.operationPorts().size(), "operation ports"))
    return std::move(error);
  for (const MemoryOperationPortDispatchDeclaration &port :
       record.operationPorts()) {
    if (llvm::Error error = writer.count(port.capabilityTargetDomains.size(),
                                         "capability target domains"))
      return std::move(error);
    for (llvm::ArrayRef<MemoryDispatchTarget> targets :
         port.capabilityTargetDomains)
      if (llvm::Error error = appendTargets(writer, targets))
        return std::move(error);
  }

  if (llvm::Error error = writer.count(record.subordinateEndpoints().size(),
                                       "subordinate endpoints"))
    return std::move(error);
  for (const MemorySubordinateDispatchDeclaration &subordinate :
       record.subordinateEndpoints()) {
    writer.u64(subordinate.maxExposedBindings);
    if (llvm::Error error =
            writer.count(subordinate.matchFields.size(), "match fields"))
      return std::move(error);
    for (MemoryProviderMatchField field : subordinate.matchFields)
      writer.u32(static_cast<std::uint32_t>(field));
    writer.u32(static_cast<std::uint32_t>(subordinate.addressTransform));
    if (llvm::Error error = appendTargets(writer, subordinate.targetDomain))
      return std::move(error);
  }

  if (llvm::Error error = writer.count(record.internalConnections().size(),
                                       "internal connections"))
    return std::move(error);
  for (const MemoryInternalConnectionDeclaration &connection :
       record.internalConnections()) {
    writer.u64(connection.sourceEndpointOrdinal);
    writer.u64(connection.sinkEndpointOrdinal);
  }
  return writer.take();
}

llvm::Expected<MemoryConnectivityContractRecord>
decodeMemoryConnectivityContractRecord(llvm::ArrayRef<std::uint8_t> bytes) {
  Reader reader(bytes);
  auto declaration = readDeclaration(reader);
  if (!declaration)
    return declaration.takeError();
  if (llvm::Error error = reader.finish())
    return std::move(error);
  auto record =
      MemoryConnectivityContractRecord::fromCanonical(std::move(*declaration));
  if (!record)
    return record.takeError();
  auto canonical = encodeMemoryConnectivityContractRecord(*record);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*canonical) != bytes)
    return invalid("record is not canonical");
  return record;
}

} // namespace fabric
