#include "SystemPnrSearchDomainInternal.h"

#include "SystemPnrDerivedContextInternal.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ComponentViewDigest.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Mapping/Artifact/MappingArtifact.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace loom::pnr {

using ::loom::mapping::SystemPresburgerCell;

namespace detail {
struct SystemPnrSearchDomainViewBuilder final {
  static SystemPnrSearchDomainView
  create(ArtifactRootReference dataflowReference,
         ArtifactRootReference fabricReference,
         ArtifactRootReference constraintReference,
         std::vector<::dataflow::RootThreadLaunchRef> rootThreadLaunches,
         std::vector<SystemSearchBindingDomain> bindings,
         std::vector<SystemSearchServiceDomain> serviceObligations,
         std::vector<std::uint8_t> canonicalViewBytes,
         SystemPnrSearchDomainDigest digest) {
    return SystemPnrSearchDomainView(
        std::move(dataflowReference), std::move(fabricReference),
        std::move(constraintReference), std::move(rootThreadLaunches),
        std::move(bindings), std::move(serviceObligations),
        std::move(canonicalViewBytes), std::move(digest));
  }
};
} // namespace detail

namespace {

constexpr char kSchemaDescriptor[] = "loom.system_pnr_search_domain.4.0";
constexpr char kDigestDomain[] = "loom.system.pnr.search.domain.digest.v1\0";

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_pnr_search_domain_invalid: " +
                                     message);
}

class WireWriter final {
public:
  void u32(std::uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
  }
  void u64(std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
  }
  void i64(std::int64_t value) { u64(static_cast<std::uint64_t>(value)); }
  void bytes(llvm::ArrayRef<std::uint8_t> value) {
    bytes_.insert(bytes_.end(), value.begin(), value.end());
  }
  void sizedBytes(llvm::ArrayRef<std::uint8_t> value) {
    u64(value.size());
    bytes(value);
  }
  void rootReference(const ArtifactRootReference &reference) {
    bytes(encodeArtifactRootReference(reference));
  }
  std::vector<std::uint8_t> take() { return std::move(bytes_); }

private:
  std::vector<std::uint8_t> bytes_;
};

class WireReader final {
public:
  explicit WireReader(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<std::uint32_t> u32() {
    if (bytes_.size() < 4)
      return invalid("truncated u32 field");
    std::uint32_t value = 0;
    for (unsigned index = 0; index < 4; ++index)
      value = (value << 8) | bytes_[index];
    bytes_ = bytes_.drop_front(4);
    return value;
  }

  llvm::Expected<std::uint64_t> u64() {
    if (bytes_.size() < 8)
      return invalid("truncated u64 field");
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index)
      value = (value << 8) | bytes_[index];
    bytes_ = bytes_.drop_front(8);
    return value;
  }

  llvm::Expected<std::int64_t> i64() {
    auto value = u64();
    if (!value)
      return value.takeError();
    return static_cast<std::int64_t>(*value);
  }

  llvm::Expected<llvm::ArrayRef<std::uint8_t>> take(std::uint64_t size) {
    if (size > bytes_.size())
      return invalid("truncated byte field");
    llvm::ArrayRef<std::uint8_t> value = bytes_.take_front(size);
    bytes_ = bytes_.drop_front(size);
    return value;
  }

  llvm::Expected<llvm::ArrayRef<std::uint8_t>> sizedBytes() {
    auto size = u64();
    if (!size)
      return size.takeError();
    return take(*size);
  }

  llvm::Expected<ArtifactRootReference> rootReference() {
    auto decoded = decodeArtifactRootReferencePrefix(bytes_);
    if (!decoded)
      return decoded.takeError();
    bytes_ = bytes_.drop_front(decoded->byteCount);
    return std::move(decoded->reference);
  }

  llvm::Expected<std::size_t> count(std::size_t minimumElementBytes,
                                    const llvm::Twine &what) {
    auto value = u64();
    if (!value)
      return value.takeError();
    if (*value > std::numeric_limits<std::size_t>::max() ||
        (minimumElementBytes != 0 &&
         *value > bytes_.size() / minimumElementBytes))
      return invalid(what + " count exceeds remaining bytes");
    return static_cast<std::size_t>(*value);
  }

  bool empty() const { return bytes_.empty(); }
  llvm::ArrayRef<std::uint8_t> remaining() const { return bytes_; }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
};

ArtifactRootReference dataflowRootReference(
    const ::dataflow::CanonicalDataflowProgramView &dataflow) {
  return {::dataflow::canonicalDataflowSchema.identity.str(),
          ::dataflow::canonicalDataflowSchema.version, dataflow.identity()};
}

ArtifactRootReference
fabricRootReference(const ::loom::fabric::FabricSystemRootView &fabric) {
  return {::loom::fabric::fabricArtifactSchema.identity.str(),
          ::loom::fabric::fabricArtifactSchema.version,
          fabric.artifact().identity()};
}

void encodeCell(WireWriter &writer, const SystemPresburgerCell &cell) {
  writer.u32(cell.dimensionCount);
  writer.u32(cell.symbolCount);
  writer.u32(cell.localCount);
  writer.u64(cell.equalities.size());
  for (const std::vector<std::int64_t> &row : cell.equalities)
    for (std::int64_t value : row)
      writer.i64(value);
  writer.u64(cell.inequalities.size());
  for (const std::vector<std::int64_t> &row : cell.inequalities)
    for (std::int64_t value : row)
      writer.i64(value);
}

llvm::Expected<SystemPresburgerCell> decodeCell(WireReader &reader) {
  auto dimensions = reader.u32();
  if (!dimensions)
    return dimensions.takeError();
  auto symbols = reader.u32();
  if (!symbols)
    return symbols.takeError();
  auto locals = reader.u32();
  if (!locals)
    return locals.takeError();
  const std::uint64_t rowWidth =
      static_cast<std::uint64_t>(*dimensions) + *symbols + *locals + 1;
  if (rowWidth > std::numeric_limits<std::size_t>::max() / 8)
    return invalid("Presburger row width exceeds native range");
  const std::size_t minimumRowBytes = static_cast<std::size_t>(rowWidth) * 8;

  SystemPresburgerCell cell;
  cell.dimensionCount = *dimensions;
  cell.symbolCount = *symbols;
  cell.localCount = *locals;
  auto equalities = reader.count(minimumRowBytes, "equality row");
  if (!equalities)
    return equalities.takeError();
  cell.equalities.resize(*equalities);
  for (std::vector<std::int64_t> &row : cell.equalities) {
    row.reserve(rowWidth);
    for (std::uint64_t column = 0; column < rowWidth; ++column) {
      auto value = reader.i64();
      if (!value)
        return value.takeError();
      row.push_back(*value);
    }
  }
  auto inequalities = reader.count(minimumRowBytes, "inequality row");
  if (!inequalities)
    return inequalities.takeError();
  cell.inequalities.resize(*inequalities);
  for (std::vector<std::int64_t> &row : cell.inequalities) {
    row.reserve(rowWidth);
    for (std::uint64_t column = 0; column < rowWidth; ++column) {
      auto value = reader.i64();
      if (!value)
        return value.takeError();
      row.push_back(*value);
    }
  }
  return cell;
}

template <typename Ref>
llvm::Error encodeFabricDomain(WireWriter &writer, llvm::ArrayRef<Ref> values) {
  writer.u64(values.size());
  for (const Ref &value : values)
    writer.sizedBytes(::loom::fabric::canonicalFabricBytes(value));
  return llvm::Error::success();
}

template <typename Ref>
void canonicalizeFabricDomain(std::vector<Ref> &values) {
  llvm::sort(values, [](const Ref &left, const Ref &right) {
    return ::loom::fabric::canonicalFabricBytes(left) <
           ::loom::fabric::canonicalFabricBytes(right);
  });
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

void encodeRootDomain(WireWriter &writer,
                      llvm::ArrayRef<ArtifactRootReference> values) {
  writer.u64(values.size());
  for (const ArtifactRootReference &value : values)
    writer.rootReference(value);
}

template <typename Ref>
llvm::Expected<std::vector<Ref>>
decodeFabricDomain(WireReader &reader,
                   const ::loom::fabric::FabricArtifactView &fabric) {
  auto count = reader.count(/*minimumElementBytes=*/8, "Fabric target");
  if (!count)
    return count.takeError();
  std::vector<Ref> values;
  values.reserve(*count);
  std::vector<std::uint8_t> previous;
  for (std::size_t index = 0; index < *count; ++index) {
    auto bytes = reader.sizedBytes();
    if (!bytes)
      return bytes.takeError();
    auto decoded = ::loom::fabric::decodeFabricRef<Ref>(*bytes);
    if (!decoded)
      return decoded.takeError();
    const std::vector<std::uint8_t> canonical =
        ::loom::fabric::canonicalFabricBytes(*decoded);
    if (llvm::ArrayRef(canonical) != *bytes)
      return invalid("noncanonical Fabric target reference");
    if (llvm::Error error = ::loom::fabric::validateFabricRef(fabric, *decoded))
      return std::move(error);
    std::vector<std::uint8_t> key(bytes->begin(), bytes->end());
    if (!previous.empty() && !(previous < key))
      return invalid("Fabric target domain is not strictly ordered");
    previous = std::move(key);
    values.push_back(std::move(*decoded));
  }
  return values;
}

llvm::Expected<std::vector<ArtifactRootReference>>
decodeRootDomain(WireReader &reader) {
  auto count = reader.count(/*minimumElementBytes=*/44, "Artifact target");
  if (!count)
    return count.takeError();
  std::vector<ArtifactRootReference> values;
  values.reserve(*count);
  for (std::size_t index = 0; index < *count; ++index) {
    auto reference = reader.rootReference();
    if (!reference)
      return reference.takeError();
    if (!values.empty() &&
        !artifactRootReferenceLess(values.back(), *reference))
      return invalid("Artifact target domain is not strictly ordered");
    values.push_back(std::move(*reference));
  }
  return values;
}

llvm::Error encodeAtomDomain(WireWriter &writer,
                             const SystemSearchAtomDomain &domain) {
  WireWriter payload;
  if (const auto *thread = std::get_if<SystemThreadBindingDomain>(&domain)) {
    writer.u32(0);
    if (llvm::Error error = encodeFabricDomain(
            payload, llvm::ArrayRef(thread->compatibleAccCores)))
      return error;
  } else if (const auto *hierarchical =
                 std::get_if<SystemHierarchicalGraphBindingDomain>(&domain)) {
    writer.u32(1);
    encodeRootDomain(payload, hierarchical->compatibleSpatialMappings);
  }
  writer.sizedBytes(payload.take());
  return llvm::Error::success();
}

llvm::Expected<SystemSearchAtomDomain>
decodeAtomDomain(WireReader &reader,
                 const ::loom::fabric::FabricArtifactView &fabric) {
  auto kind = reader.u32();
  if (!kind)
    return kind.takeError();
  auto payloadBytes = reader.sizedBytes();
  if (!payloadBytes)
    return payloadBytes.takeError();
  WireReader payload(*payloadBytes);
  SystemSearchAtomDomain result;
  if (*kind == 0) {
    auto values = decodeFabricDomain<::loom::fabric::AccCoreOccurrenceRef>(
        payload, fabric);
    if (!values)
      return values.takeError();
    result = SystemThreadBindingDomain{std::move(*values)};
  } else if (*kind == 1) {
    auto values = decodeRootDomain(payload);
    if (!values)
      return values.takeError();
    result = SystemHierarchicalGraphBindingDomain{std::move(*values)};
  } else {
    return invalid("unknown System search-atom domain variant");
  }
  if (!payload.empty())
    return invalid("System search-atom domain has trailing bytes");
  return result;
}

template <typename Ref>
llvm::Expected<Ref>
decodeDataflowKey(WireReader &reader,
                  const ArtifactIdentity &dataflowIdentity) {
  auto bytes = reader.sizedBytes();
  if (!bytes)
    return bytes.takeError();
  auto decoded =
      ::dataflow::decodeDataflowReference<Ref>(*bytes, dataflowIdentity);
  if (!decoded)
    return decoded.takeError();
  auto canonical =
      ::dataflow::encodeDataflowReference(dataflowIdentity, *decoded);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef(*canonical) != *bytes)
    return invalid("noncanonical Dataflow binding reference");
  return *decoded;
}

llvm::Expected<SystemSearchBindingKey>
decodeBindingKey(WireReader &reader, const ArtifactIdentity &dataflowIdentity) {
  auto kind = reader.u32();
  if (!kind)
    return kind.takeError();
  if (*kind == 0) {
    auto ref = decodeDataflowKey<::dataflow::RootThreadLaunchRef>(
        reader, dataflowIdentity);
    if (!ref)
      return ref.takeError();
    return SystemSearchBindingKey(*ref);
  }
  if (*kind == 1) {
    auto ref = decodeDataflowKey<::dataflow::RootedGraphLaunchRef>(
        reader, dataflowIdentity);
    if (!ref)
      return ref.takeError();
    return SystemSearchBindingKey(*ref);
  }
  return invalid("unknown binding-key kind");
}

llvm::Error encodeBindingKey(WireWriter &writer,
                             const SystemSearchBindingKey &key,
                             const ArtifactIdentity &dataflowIdentity) {
  if (const auto *thread = std::get_if<::dataflow::RootThreadLaunchRef>(&key)) {
    writer.u32(0);
    auto bytes = ::dataflow::encodeDataflowReference(dataflowIdentity, *thread);
    if (!bytes)
      return bytes.takeError();
    writer.sizedBytes(*bytes);
    return llvm::Error::success();
  }
  writer.u32(1);
  auto bytes = ::dataflow::encodeDataflowReference(
      dataflowIdentity, std::get<::dataflow::RootedGraphLaunchRef>(key));
  if (!bytes)
    return bytes.takeError();
  writer.sizedBytes(*bytes);
  return llvm::Error::success();
}

llvm::Error
encodeTerminalKey(WireWriter &writer,
                  const ::loom::mapping::SystemTransferTerminalKey &key,
                  const ArtifactIdentity &dataflowIdentity) {
  auto bytes =
      ::loom::mapping::encodeSystemTransferTerminalKey(dataflowIdentity, key);
  if (!bytes)
    return bytes.takeError();
  writer.bytes(*bytes);
  return llvm::Error::success();
}

llvm::Expected<std::vector<std::uint8_t>>
terminalKeyBytes(const ::loom::mapping::SystemTransferTerminalKey &key,
                 const ArtifactIdentity &dataflowIdentity) {
  return ::loom::mapping::encodeSystemTransferTerminalKey(dataflowIdentity,
                                                          key);
}

llvm::Expected<::loom::mapping::SystemTransferTerminalKey>
decodeTerminalKey(WireReader &reader,
                  const ArtifactIdentity &dataflowIdentity) {
  auto decoded = ::loom::mapping::decodeSystemTransferTerminalKeyPrefix(
      reader.remaining(), dataflowIdentity);
  if (!decoded)
    return decoded.takeError();
  auto consumed = reader.take(decoded->byteCount);
  if (!consumed)
    return consumed.takeError();
  return std::move(decoded->key);
}

template <typename Ref>
llvm::Expected<Ref>
decodeDataflowPayload(llvm::ArrayRef<std::uint8_t> bytes,
                      const ArtifactIdentity &dataflowIdentity) {
  auto decoded =
      ::dataflow::decodeDataflowReference<Ref>(bytes, dataflowIdentity);
  if (!decoded)
    return decoded.takeError();
  auto canonical =
      ::dataflow::encodeDataflowReference(dataflowIdentity, *decoded);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef(*canonical) != bytes)
    return invalid("nested Dataflow reference is not canonical");
  return std::move(*decoded);
}

llvm::Expected<std::vector<std::uint8_t>>
encodeTargetSubjectBytes(const SystemServiceTargetSubject &subject,
                         const ArtifactIdentity &dataflowIdentity) {
  WireWriter writer;
  if (const auto *member =
          std::get_if<SystemServiceMemberTargetSubject>(&subject)) {
    writer.u32(0);
    auto bytes =
        ::dataflow::encodeDataflowReference(dataflowIdentity, member->member);
    if (!bytes)
      return bytes.takeError();
    writer.sizedBytes(*bytes);
  } else {
    writer.u32(1);
    auto bytes = ::dataflow::encodeDataflowReference(
        dataflowIdentity,
        std::get<SystemMemoryExposureTargetSubject>(subject).exposure);
    if (!bytes)
      return bytes.takeError();
    writer.sizedBytes(*bytes);
  }
  return writer.take();
}

llvm::Expected<SystemServiceTargetSubject>
decodeTargetSubject(WireReader &reader,
                    const ArtifactIdentity &dataflowIdentity) {
  auto kind = reader.u32();
  if (!kind)
    return kind.takeError();
  auto payload = reader.sizedBytes();
  if (!payload)
    return payload.takeError();
  if (*kind == 0) {
    auto member = decodeDataflowPayload<::dataflow::ServiceMemberRef>(
        *payload, dataflowIdentity);
    if (!member)
      return member.takeError();
    if (std::holds_alternative<::dataflow::MessageTransferMemberRef>(*member))
      return invalid("message member cannot be a service target subject");
    return SystemServiceTargetSubject{
        SystemServiceMemberTargetSubject{std::move(*member)}};
  }
  if (*kind == 1) {
    auto exposure = decodeDataflowPayload<::dataflow::MemoryExposureRef>(
        *payload, dataflowIdentity);
    if (!exposure)
      return exposure.takeError();
    return SystemServiceTargetSubject{
        SystemMemoryExposureTargetSubject{std::move(*exposure)}};
  }
  return invalid("unknown service-target subject variant");
}

llvm::Expected<std::vector<std::uint8_t>>
encodeBoundEndpointBytes(const SystemBoundTerminalEndpoint &endpoint) {
  WireWriter writer;
  if (const auto *message =
          std::get_if<SystemMessageTerminalEndpoint>(&endpoint)) {
    writer.u32(0);
    writer.sizedBytes(::loom::fabric::canonicalFabricBytes(message->endpoint));
  } else {
    writer.u32(1);
    writer.sizedBytes(::loom::fabric::canonicalFabricBytes(
        std::get<SystemMemoryOrFenceTerminalEndpoint>(endpoint).endpoint));
  }
  return writer.take();
}

llvm::Expected<SystemBoundTerminalEndpoint>
decodeBoundEndpoint(WireReader &reader,
                    const ::loom::fabric::FabricArtifactView &fabric) {
  auto kind = reader.u32();
  if (!kind)
    return kind.takeError();
  auto payload = reader.sizedBytes();
  if (!payload)
    return payload.takeError();
  if (*kind == 0) {
    auto endpoint = ::loom::fabric::decodeFabricRef<
        ::loom::fabric::FabricTransportEndpointRef>(*payload);
    if (!endpoint)
      return endpoint.takeError();
    if (llvm::ArrayRef(::loom::fabric::canonicalFabricBytes(*endpoint)) !=
        *payload)
      return invalid("message terminal endpoint is not canonical");
    if (llvm::Error error =
            ::loom::fabric::validateFabricRef(fabric, *endpoint))
      return std::move(error);
    return SystemBoundTerminalEndpoint{
        SystemMessageTerminalEndpoint{std::move(*endpoint)}};
  }
  if (*kind == 1) {
    auto endpoint = ::loom::fabric::decodeFabricRef<
        ::loom::fabric::FabricMemoryEndpointRef>(*payload);
    if (!endpoint)
      return endpoint.takeError();
    if (llvm::ArrayRef(::loom::fabric::canonicalFabricBytes(*endpoint)) !=
        *payload)
      return invalid("memory terminal endpoint is not canonical");
    if (llvm::Error error =
            ::loom::fabric::validateFabricRef(fabric, *endpoint))
      return std::move(error);
    return SystemBoundTerminalEndpoint{
        SystemMemoryOrFenceTerminalEndpoint{std::move(*endpoint)}};
  }
  return invalid("unknown bound-terminal endpoint variant");
}

const ::loom::mapping::CanonicalServiceLegKey &
terminalLeg(const ::loom::mapping::SystemTransferTerminalKey &terminal) {
  if (const auto *source =
          std::get_if<::loom::mapping::SystemTransferSourceTerminalKey>(
              &terminal))
    return source->leg;
  return std::get<::loom::mapping::SystemTransferSinkTerminalKey>(terminal).leg;
}

llvm::Error
canonicalizeServiceDomains(std::vector<SystemSearchServiceDomain> &services,
                           const ArtifactIdentity &dataflowIdentity) {
  std::vector<std::pair<std::vector<std::uint8_t>, SystemSearchServiceDomain>>
      ordered;
  ordered.reserve(services.size());
  for (SystemSearchServiceDomain &service : services) {
    std::vector<std::uint8_t> previous;
    std::vector<std::pair<std::vector<std::uint8_t>,
                          SystemSearchServiceTargetCompatibility>>
        targets;
    targets.reserve(service.targetCompatibility.size());
    for (SystemSearchServiceTargetCompatibility &target :
         service.targetCompatibility) {
      if (auto *regions = std::get_if<
              std::vector<::loom::fabric::FabricMemoryServiceRegionRef>>(
              &target.compatibleTargets))
        canonicalizeFabricDomain(*regions);
      else
        canonicalizeFabricDomain(
            std::get<std::vector<::loom::fabric::MemoryConsistencyDomainRef>>(
                target.compatibleTargets));
      auto subject = encodeTargetSubjectBytes(target.subject, dataflowIdentity);
      if (!subject)
        return subject.takeError();
      WireWriter key;
      key.bytes(*subject);
      key.sizedBytes(
          ::loom::fabric::canonicalFabricBytes(target.boundEndpoint));
      targets.emplace_back(key.take(), std::move(target));
    }
    llvm::sort(targets, [](const auto &lhs, const auto &rhs) {
      return lhs.first < rhs.first;
    });
    service.targetCompatibility.clear();
    previous.clear();
    for (auto &entry : targets) {
      if (!previous.empty() && previous == entry.first)
        return invalid("service target-compatibility row is duplicated");
      previous = entry.first;
      service.targetCompatibility.push_back(std::move(entry.second));
    }

    std::vector<std::pair<std::vector<std::uint8_t>,
                          SystemSearchTransferTerminalCompatibility>>
        terminals;
    terminals.reserve(service.transferTerminalCompatibility.size());
    for (SystemSearchTransferTerminalCompatibility &terminal :
         service.transferTerminalCompatibility) {
      canonicalizeFabricDomain(terminal.compatibleTransportEndpoints);
      if (terminalLeg(terminal.terminal).obligation != service.key)
        return invalid("transfer-terminal key belongs to another obligation");
      auto terminalBytes =
          terminalKeyBytes(terminal.terminal, dataflowIdentity);
      if (!terminalBytes)
        return terminalBytes.takeError();
      auto endpointBytes = encodeBoundEndpointBytes(terminal.boundEndpoint);
      if (!endpointBytes)
        return endpointBytes.takeError();
      WireWriter key;
      key.bytes(*terminalBytes);
      key.bytes(*endpointBytes);
      terminals.emplace_back(key.take(), std::move(terminal));
    }
    llvm::sort(terminals, [](const auto &left, const auto &right) {
      return left.first < right.first;
    });
    service.transferTerminalCompatibility.clear();
    previous.clear();
    for (auto &entry : terminals) {
      if (!previous.empty() && previous == entry.first)
        return invalid("transfer-terminal compatibility row is duplicated");
      previous = entry.first;
      service.transferTerminalCompatibility.push_back(std::move(entry.second));
    }

    auto bytes = ::loom::mapping::encodeSystemServiceObligationKey(
        dataflowIdentity, service.key);
    if (!bytes)
      return bytes.takeError();
    ordered.emplace_back(std::move(*bytes), std::move(service));
  }
  llvm::sort(ordered, [](const auto &left, const auto &right) {
    return left.first < right.first;
  });
  services.clear();
  std::vector<std::uint8_t> previousService;
  for (auto &entry : ordered) {
    if (!previousService.empty() && previousService == entry.first)
      return invalid("service-obligation key is duplicated");
    previousService = entry.first;
    services.push_back(std::move(entry.second));
  }
  return llvm::Error::success();
}

llvm::Expected<std::vector<std::uint8_t>>
encodeView(const ArtifactRootReference &dataflowReference,
           const ArtifactRootReference &fabricReference,
           const ArtifactRootReference &constraintReference,
           llvm::ArrayRef<::dataflow::RootThreadLaunchRef> roots,
           llvm::ArrayRef<SystemSearchBindingDomain> bindings,
           llvm::ArrayRef<SystemSearchServiceDomain> services) {
  WireWriter writer;
  writer.rootReference(dataflowReference);
  writer.rootReference(fabricReference);
  writer.rootReference(constraintReference);
  writer.u64(roots.size());
  for (::dataflow::RootThreadLaunchRef root : roots) {
    auto bytes =
        ::dataflow::encodeDataflowReference(dataflowReference.artifact, root);
    if (!bytes)
      return bytes.takeError();
    writer.sizedBytes(*bytes);
  }
  writer.u64(bindings.size());
  for (const SystemSearchBindingDomain &binding : bindings) {
    if (llvm::Error error =
            encodeBindingKey(writer, binding.key, dataflowReference.artifact))
      return std::move(error);
    writer.u64(binding.atoms.size());
    for (const SystemSearchAtom &atom : binding.atoms) {
      encodeCell(writer, atom.cell);
      if (llvm::Error error = encodeAtomDomain(writer, atom.domain))
        return std::move(error);
    }
  }
  writer.u64(services.size());
  for (const SystemSearchServiceDomain &service : services) {
    auto keyBytes = ::loom::mapping::encodeSystemServiceObligationKey(
        dataflowReference.artifact, service.key);
    if (!keyBytes)
      return keyBytes.takeError();
    writer.sizedBytes(*keyBytes);
    writer.u64(service.targetCompatibility.size());
    for (const SystemSearchServiceTargetCompatibility &target :
         service.targetCompatibility) {
      auto subject =
          encodeTargetSubjectBytes(target.subject, dataflowReference.artifact);
      if (!subject)
        return subject.takeError();
      writer.sizedBytes(*subject);
      writer.sizedBytes(
          ::loom::fabric::canonicalFabricBytes(target.boundEndpoint));
      WireWriter payload;
      if (const auto *regions = std::get_if<
              std::vector<::loom::fabric::FabricMemoryServiceRegionRef>>(
              &target.compatibleTargets)) {
        writer.u32(0);
        if (llvm::Error error =
                encodeFabricDomain(payload, llvm::ArrayRef(*regions)))
          return std::move(error);
      } else {
        writer.u32(1);
        if (llvm::Error error = encodeFabricDomain(
                payload,
                llvm::ArrayRef(std::get<std::vector<
                                   ::loom::fabric::MemoryConsistencyDomainRef>>(
                    target.compatibleTargets))))
          return std::move(error);
      }
      writer.sizedBytes(payload.take());
    }
    writer.u64(service.transferTerminalCompatibility.size());
    for (const SystemSearchTransferTerminalCompatibility &terminal :
         service.transferTerminalCompatibility) {
      if (llvm::Error error = encodeTerminalKey(writer, terminal.terminal,
                                                dataflowReference.artifact))
        return std::move(error);
      auto endpoint = encodeBoundEndpointBytes(terminal.boundEndpoint);
      if (!endpoint)
        return endpoint.takeError();
      writer.bytes(*endpoint);
      if (llvm::Error error = encodeFabricDomain(
              writer, llvm::ArrayRef(terminal.compatibleTransportEndpoints)))
        return std::move(error);
    }
  }
  return writer.take();
}

struct DecodedView final {
  ArtifactRootReference dataflowReference;
  ArtifactRootReference fabricReference;
  ArtifactRootReference constraintReference;
  std::vector<::dataflow::RootThreadLaunchRef> roots;
  std::vector<SystemSearchBindingDomain> bindings;
  std::vector<SystemSearchServiceDomain> services;
};

llvm::Expected<DecodedView>
decodeView(llvm::ArrayRef<std::uint8_t> bytes,
           const ::loom::fabric::FabricArtifactView &fabric) {
  WireReader reader(bytes);
  auto dataflowReference = reader.rootReference();
  if (!dataflowReference)
    return dataflowReference.takeError();
  auto fabricReference = reader.rootReference();
  if (!fabricReference)
    return fabricReference.takeError();
  auto constraintReference = reader.rootReference();
  if (!constraintReference)
    return constraintReference.takeError();

  auto rootCount = reader.count(/*minimumElementBytes=*/16, "root launch");
  if (!rootCount)
    return rootCount.takeError();
  std::vector<::dataflow::RootThreadLaunchRef> roots;
  roots.reserve(*rootCount);
  for (std::size_t index = 0; index < *rootCount; ++index) {
    auto root = decodeDataflowKey<::dataflow::RootThreadLaunchRef>(
        reader, dataflowReference->artifact);
    if (!root)
      return root.takeError();
    roots.push_back(*root);
  }

  auto bindingCount = reader.count(/*minimumElementBytes=*/20, "binding");
  if (!bindingCount)
    return bindingCount.takeError();
  std::vector<SystemSearchBindingDomain> bindings;
  bindings.reserve(*bindingCount);
  for (std::size_t index = 0; index < *bindingCount; ++index) {
    auto key = decodeBindingKey(reader, dataflowReference->artifact);
    if (!key)
      return key.takeError();
    auto atomCount = reader.count(/*minimumElementBytes=*/20, "binding atom");
    if (!atomCount)
      return atomCount.takeError();
    SystemSearchBindingDomain binding{std::move(*key), {}};
    binding.atoms.reserve(*atomCount);
    for (std::size_t atomIndex = 0; atomIndex < *atomCount; ++atomIndex) {
      auto cell = decodeCell(reader);
      if (!cell)
        return cell.takeError();
      auto domain = decodeAtomDomain(reader, fabric);
      if (!domain)
        return domain.takeError();
      binding.atoms.push_back({std::move(*cell), std::move(*domain)});
    }
    bindings.push_back(std::move(binding));
  }

  auto serviceCount =
      reader.count(/*minimumElementBytes=*/20, "service obligation");
  if (!serviceCount)
    return serviceCount.takeError();
  std::vector<SystemSearchServiceDomain> services;
  services.reserve(*serviceCount);
  std::vector<std::uint8_t> previousService;
  for (std::size_t index = 0; index < *serviceCount; ++index) {
    auto keyBytes = reader.sizedBytes();
    if (!keyBytes)
      return keyBytes.takeError();
    auto key = ::loom::mapping::decodeSystemServiceObligationKey(
        *keyBytes, dataflowReference->artifact);
    if (!key)
      return key.takeError();
    std::vector<std::uint8_t> keyStorage(keyBytes->begin(), keyBytes->end());
    if (!previousService.empty() && !(previousService < keyStorage))
      return invalid("service-obligation domains are not strictly ordered");
    previousService = keyStorage;

    SystemSearchServiceDomain service{std::move(*key), {}, {}};

    auto targetCount =
        reader.count(/*minimumElementBytes=*/28, "target compatibility");
    if (!targetCount)
      return targetCount.takeError();
    std::vector<std::uint8_t> previousTarget;
    service.targetCompatibility.reserve(*targetCount);
    for (std::size_t targetIndex = 0; targetIndex < *targetCount;
         ++targetIndex) {
      auto subjectBytes = reader.sizedBytes();
      if (!subjectBytes)
        return subjectBytes.takeError();
      WireReader subjectReader(*subjectBytes);
      auto subject =
          decodeTargetSubject(subjectReader, dataflowReference->artifact);
      if (!subject)
        return subject.takeError();
      if (!subjectReader.empty())
        return invalid("target-compatibility subject has trailing bytes");
      auto endpointBytes = reader.sizedBytes();
      if (!endpointBytes)
        return endpointBytes.takeError();
      auto endpoint = ::loom::fabric::decodeFabricRef<
          ::loom::fabric::SystemServiceEndpointRef>(*endpointBytes);
      if (!endpoint)
        return endpoint.takeError();
      if (llvm::ArrayRef(::loom::fabric::canonicalFabricBytes(*endpoint)) !=
          *endpointBytes)
        return invalid("bound System service endpoint is not canonical");
      if (llvm::Error error =
              ::loom::fabric::validateFabricRef(fabric, *endpoint))
        return std::move(error);
      WireWriter targetKey;
      targetKey.bytes(*subjectBytes);
      targetKey.sizedBytes(*endpointBytes);
      std::vector<std::uint8_t> rowKey = targetKey.take();
      if (!previousTarget.empty() && !(previousTarget < rowKey))
        return invalid("target-compatibility rows are not strictly ordered");
      previousTarget = std::move(rowKey);
      auto targetKind = reader.u32();
      if (!targetKind)
        return targetKind.takeError();
      auto domainBytes = reader.sizedBytes();
      if (!domainBytes)
        return domainBytes.takeError();
      WireReader domainReader(*domainBytes);
      SystemServiceTargetCompatibilityDomain compatible;
      if (*targetKind == 0) {
        auto values =
            decodeFabricDomain<::loom::fabric::FabricMemoryServiceRegionRef>(
                domainReader, fabric);
        if (!values)
          return values.takeError();
        compatible = std::move(*values);
      } else if (*targetKind == 1) {
        auto values =
            decodeFabricDomain<::loom::fabric::MemoryConsistencyDomainRef>(
                domainReader, fabric);
        if (!values)
          return values.takeError();
        compatible = std::move(*values);
      } else {
        return invalid("unknown target-compatibility domain variant");
      }
      if (!domainReader.empty())
        return invalid("target-compatibility domain has trailing bytes");
      service.targetCompatibility.push_back(
          {std::move(*subject), std::move(*endpoint), std::move(compatible)});
    }

    auto terminalCount = reader.count(/*minimumElementBytes=*/32,
                                      "transfer terminal compatibility");
    if (!terminalCount)
      return terminalCount.takeError();
    std::vector<std::uint8_t> previousTerminal;
    service.transferTerminalCompatibility.reserve(*terminalCount);
    for (std::size_t terminalIndex = 0; terminalIndex < *terminalCount;
         ++terminalIndex) {
      auto terminalKey = decodeTerminalKey(reader, dataflowReference->artifact);
      if (!terminalKey)
        return terminalKey.takeError();
      if (terminalLeg(*terminalKey).obligation != service.key)
        return invalid("transfer-terminal key belongs to another obligation");
      auto terminalBytes =
          terminalKeyBytes(*terminalKey, dataflowReference->artifact);
      if (!terminalBytes)
        return terminalBytes.takeError();
      auto boundEndpoint = decodeBoundEndpoint(reader, fabric);
      if (!boundEndpoint)
        return boundEndpoint.takeError();
      auto boundBytes = encodeBoundEndpointBytes(*boundEndpoint);
      if (!boundBytes)
        return boundBytes.takeError();
      WireWriter rowKeyWriter;
      rowKeyWriter.bytes(*terminalBytes);
      rowKeyWriter.bytes(*boundBytes);
      std::vector<std::uint8_t> rowKey = rowKeyWriter.take();
      if (!previousTerminal.empty() && !(previousTerminal < rowKey))
        return invalid(
            "transfer-terminal compatibility rows are not strictly ordered");
      previousTerminal = std::move(rowKey);
      auto endpoints =
          decodeFabricDomain<::loom::fabric::FabricTransportEndpointRef>(
              reader, fabric);
      if (!endpoints)
        return endpoints.takeError();
      service.transferTerminalCompatibility.push_back(
          {std::move(*terminalKey), std::move(*boundEndpoint),
           std::move(*endpoints)});
    }
    services.push_back(std::move(service));
  }
  if (!reader.empty())
    return invalid("trailing canonical view bytes");
  return DecodedView{std::move(*dataflowReference),
                     std::move(*fabricReference),
                     std::move(*constraintReference),
                     std::move(roots),
                     std::move(bindings),
                     std::move(services)};
}

llvm::Error validateConstraintInputs(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const ::loom::mapping::FinalizedSystemMappingConstraintSet &constraints,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> canonicalRoots,
    llvm::ArrayRef<ArtifactRootReference> spatialMappings) {
  if (constraints.view().dataflowIdentity() != dataflow.identity() ||
      constraints.view().fabricIdentity() != fabric.artifact().identity())
    return invalid("System MappingConstraintSet has foreign D/F owners");
  auto constraintRoots = detail::canonicalRootThreadLaunchSet(
      dataflow, constraints.view().rootThreadLaunches());
  if (!constraintRoots)
    return constraintRoots.takeError();
  if (llvm::ArrayRef(*constraintRoots) != canonicalRoots)
    return invalid("System MappingConstraintSet root closure differs from H");
  for (const ArtifactRootReference &required :
       constraints.view().spatialMappingReferences())
    if (!llvm::is_contained(spatialMappings, required))
      return invalid(
          "System MappingConstraintSet references a mapping outside H");
  return llvm::Error::success();
}

llvm::Expected<SystemPnrSearchDomainView> buildView(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const ResolvedPnrConfigView &config,
    const ::loom::mapping::FinalizedSystemMappingConstraintSet &constraints,
    const SystemBindingPartitionPlan &partitionPlan,
    const SystemGraphSearchInput &graphSearch, const ArtifactStore &store,
    const SystemActiveContext *activeContext) {
  if (config.domain() != PnrConfigDomain::System)
    return invalid("System search-domain projection received a non-System "
                   "resolved config view");
  if (llvm::Error error = validateComponentViewDigest(
          config.schemaDescriptorBytes(), config.canonicalViewBytes(),
          config.digest()))
    return llvm::joinErrors(invalid("System resolved config digest is invalid"),
                            std::move(error));
  auto roots = detail::canonicalRootThreadLaunchSet(
      dataflow, constraints.view().rootThreadLaunches());
  if (!roots)
    return roots.takeError();
  const llvm::ArrayRef<ArtifactRootReference> spatialMappings =
      graphSearch.spatialMappings;
  if (llvm::Error error = validateConstraintInputs(
          dataflow, fabric, constraints, *roots, spatialMappings))
    return std::move(error);
  auto constraintIndex = detail::buildFrozenConstraintIndex(constraints.view());
  if (!constraintIndex)
    return constraintIndex.takeError();
  auto partitions = detail::canonicalizeAndValidateSystemPartition(
      dataflow, *roots, partitionPlan);
  if (!partitions)
    return partitions.takeError();
  std::optional<std::vector<detail::SpatialCatalogEntry>> ownedCatalog;
  llvm::ArrayRef<detail::SpatialCatalogEntry> catalog;
  if (activeContext) {
    const auto &active = detail::systemActiveContextStorage(*activeContext);
    std::vector<ArtifactRootReference> canonicalMappings(
        spatialMappings.begin(), spatialMappings.end());
    llvm::sort(canonicalMappings, artifactRootReferenceLess);
    if (active.dataflowIdentity != dataflow.identity() ||
        active.systemIdentity != fabric.artifact().identity() ||
        active.constraintIdentity != constraints.view().identity() ||
        active.spatialMappings != canonicalMappings || !active.spatialCatalog)
      return invalid("SystemActiveContext does not match search-domain "
                     "inputs");
    catalog = *active.spatialCatalog;
  } else {
    auto imported =
        detail::importSpatialCatalog(spatialMappings, dataflow, fabric, store);
    if (!imported)
      return imported.takeError();
    ownedCatalog.emplace(std::move(*imported));
    catalog = *ownedCatalog;
  }
  std::vector<::loom::fabric::AccCoreOccurrenceRef> cores =
      detail::canonicalSystemAccCores(fabric);

  std::vector<SystemSearchBindingDomain> bindings;
  bindings.reserve(partitions->size());
  for (detail::CanonicalSystemPartitionBinding &partition : *partitions) {
    SystemSearchAtomDomain domain;
    if (std::holds_alternative<::dataflow::RootThreadLaunchRef>(
            partition.key)) {
      std::vector<::loom::fabric::AccCoreOccurrenceRef> compatible = cores;
      detail::applySystemConstraintRestriction(
          compatible, *constraintIndex,
          ::mapping::SystemConstraintProjection::ThreadTargetAccCore,
          ::loom::mapping::SystemConstraintSubject{
              std::get<::dataflow::RootThreadLaunchRef>(partition.key)});
      domain = SystemThreadBindingDomain{std::move(compatible)};
    } else {
      const auto graphLaunch =
          std::get<::dataflow::RootedGraphLaunchRef>(partition.key);
      auto graph = dataflow.resolve(graphLaunch);
      if (!graph)
        return graph.takeError();
      std::vector<ArtifactRootReference> compatible;
      for (const detail::SpatialCatalogEntry &entry : catalog)
        if (llvm::is_contained(entry.covers, *graph))
          compatible.push_back(entry.reference);
      detail::applySystemConstraintRestriction(
          compatible, *constraintIndex,
          ::mapping::SystemConstraintProjection::GraphSelectedSpatialMapping,
          ::loom::mapping::SystemConstraintSubject{graphLaunch});
      domain = SystemHierarchicalGraphBindingDomain{std::move(compatible)};
    }
    SystemSearchBindingDomain binding{std::move(partition.key), {}};
    binding.atoms.reserve(partition.cells.size());
    for (SystemPresburgerCell &cell : partition.cells)
      binding.atoms.push_back({std::move(cell), domain});
    bindings.push_back(std::move(binding));
  }

  auto services = detail::projectSystemServiceDomains(
      dataflow, fabric, *roots, bindings, catalog, *constraintIndex);
  if (!services)
    return services.takeError();
  if (llvm::Error error =
          canonicalizeServiceDomains(*services, dataflow.identity()))
    return std::move(error);

  ArtifactRootReference dataflowReference = dataflowRootReference(dataflow);
  ArtifactRootReference fabricReference = fabricRootReference(fabric);
  auto bytes = encodeView(dataflowReference, fabricReference,
                          constraints.reference(), *roots, bindings, *services);
  if (!bytes)
    return bytes.takeError();
  auto digest = computeSystemPnrSearchDomainDigest(
      systemPnrSearchDomainSchemaDescriptorBytes(), *bytes);
  if (!digest)
    return digest.takeError();
  return detail::SystemPnrSearchDomainViewBuilder::create(
      std::move(dataflowReference), std::move(fabricReference),
      constraints.reference(), std::move(*roots), std::move(bindings),
      std::move(*services), std::move(*bytes), std::move(*digest));
}

} // namespace

char UnsupportedSystemPnrSearchDomain::ID = 0;

void UnsupportedSystemPnrSearchDomain::log(llvm::raw_ostream &stream) const {
  stream << "system_pnr_search_domain_unsupported: " << message_;
}

std::error_code UnsupportedSystemPnrSearchDomain::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Expected<SystemPnrSearchDomainDigest>
SystemPnrSearchDomainDigest::fromBytes(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != byteSize)
    return invalid("digest must contain exactly 32 bytes");
  Storage storage{};
  std::copy(bytes.begin(), bytes.end(), storage.begin());
  return SystemPnrSearchDomainDigest(storage);
}

llvm::ArrayRef<std::uint8_t> systemPnrSearchDomainSchemaDescriptorBytes() {
  return {reinterpret_cast<const std::uint8_t *>(kSchemaDescriptor),
          sizeof(kSchemaDescriptor) - 1};
}

llvm::Expected<SystemPnrSearchDomainDigest> computeSystemPnrSearchDomainDigest(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes) {
  if (schemaDescriptorBytes.size() > std::numeric_limits<std::uint32_t>::max())
    return invalid("schema descriptor exceeds u32 framing");
  WireWriter writer;
  writer.bytes({reinterpret_cast<const std::uint8_t *>(kDigestDomain),
                sizeof(kDigestDomain) - 1});
  writer.u32(static_cast<std::uint32_t>(schemaDescriptorBytes.size()));
  writer.bytes(schemaDescriptorBytes);
  writer.u64(canonicalViewBytes.size());
  writer.bytes(canonicalViewBytes);
  const auto digest = llvm::SHA256::hash(writer.take());
  SystemPnrSearchDomainDigest::Storage storage{};
  std::copy(digest.begin(), digest.end(), storage.begin());
  return SystemPnrSearchDomainDigest(storage);
}

llvm::Error validateSystemPnrSearchDomainDigest(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const SystemPnrSearchDomainDigest &digest) {
  auto expected = computeSystemPnrSearchDomainDigest(schemaDescriptorBytes,
                                                     canonicalViewBytes);
  if (!expected)
    return expected.takeError();
  if (*expected != digest)
    return invalid("digest does not match canonical view bytes");
  return llvm::Error::success();
}

llvm::Expected<SystemPnrSearchDomainView> projectSystemPnrSearchDomain(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const ResolvedPnrConfigView &config,
    const ::loom::mapping::FinalizedSystemMappingConstraintSet &constraints,
    const SystemBindingPartitionPlan &partitionPlan,
    const SystemGraphSearchInput &graphSearch, const ArtifactStore &store,
    const SystemActiveContext *activeContext) {
  return buildView(dataflow, fabric, config, constraints, partitionPlan,
                   graphSearch, store, activeContext);
}

llvm::Expected<SystemPnrSearchDomainView>
adoptSystemPnrSearchDomain(llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
                           llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
                           const SystemPnrSearchDomainDigest &digest,
                           const ArtifactStore &store) {
  if (schemaDescriptorBytes != systemPnrSearchDomainSchemaDescriptorBytes())
    return invalid("schema descriptor is not exact version 4.0");
  if (llvm::Error error = validateSystemPnrSearchDomainDigest(
          schemaDescriptorBytes, canonicalViewBytes, digest))
    return std::move(error);

  WireReader prefix(canonicalViewBytes);
  auto dataflowReference = prefix.rootReference();
  if (!dataflowReference)
    return dataflowReference.takeError();
  auto fabricReference = prefix.rootReference();
  if (!fabricReference)
    return fabricReference.takeError();
  auto dataflowArtifact =
      ::dataflow::importCanonicalDataflow(*dataflowReference, store);
  if (!dataflowArtifact)
    return dataflowArtifact.takeError();
  auto dataflow = dataflowArtifact->view();
  if (!dataflow)
    return dataflow.takeError();
  auto fabricRoot =
      ::loom::fabric::importEntireFabricRoot(*fabricReference, store);
  if (!fabricRoot)
    return fabricRoot.takeError();
  auto system = ::loom::fabric::requireSystemRoot(fabricRoot->view());
  if (!system)
    return system.takeError();

  auto decoded = decodeView(canonicalViewBytes, system->artifact());
  if (!decoded)
    return decoded.takeError();
  if (decoded->dataflowReference != *dataflowReference ||
      decoded->fabricReference != *fabricReference)
    return invalid("canonical prefix owners changed during decode");
  auto constraints = ::loom::mapping::importSystemMappingConstraintSet(
      decoded->constraintReference, store);
  if (!constraints)
    return constraints.takeError();
  if (constraints->view().dataflowIdentity() != dataflow->identity() ||
      constraints->view().fabricIdentity() != system->artifact().identity())
    return invalid("adopted search domain has foreign K owners");
  auto constraintIndex =
      detail::buildFrozenConstraintIndex(constraints->view());
  if (!constraintIndex)
    return constraintIndex.takeError();
  auto roots = detail::canonicalRootThreadLaunchSet(
      *dataflow, constraints->view().rootThreadLaunches());
  if (!roots)
    return roots.takeError();
  if (llvm::ArrayRef(*roots) !=
      llvm::ArrayRef<::dataflow::RootThreadLaunchRef>(decoded->roots))
    return invalid("adopted search domain root closure differs from K");

  SystemBindingPartitionPlan plan;
  plan.bindings.reserve(decoded->bindings.size());
  for (const SystemSearchBindingDomain &binding : decoded->bindings) {
    SystemPresburgerBindingPartition partition{binding.key, {}};
    partition.cells.reserve(binding.atoms.size());
    for (const SystemSearchAtom &atom : binding.atoms)
      partition.cells.push_back(atom.cell);
    plan.bindings.push_back(std::move(partition));
  }
  auto canonicalPartitions = detail::canonicalizeAndValidateSystemPartition(
      *dataflow, decoded->roots, plan);
  if (!canonicalPartitions)
    return canonicalPartitions.takeError();
  if (canonicalPartitions->size() != decoded->bindings.size())
    return invalid("adopted binding partition is not canonical");
  for (auto &&[canonical, decodedBinding] :
       llvm::zip_equal(*canonicalPartitions, decoded->bindings)) {
    auto canonicalKey =
        detail::canonicalBindingKeyBytes(canonical.key, dataflow->identity());
    if (!canonicalKey)
      return canonicalKey.takeError();
    auto decodedKey = detail::canonicalBindingKeyBytes(decodedBinding.key,
                                                       dataflow->identity());
    if (!decodedKey)
      return decodedKey.takeError();
    if (*canonicalKey != *decodedKey ||
        canonical.cells.size() != decodedBinding.atoms.size())
      return invalid("adopted binding partition is not canonical");
    for (auto &&[cell, atom] :
         llvm::zip_equal(canonical.cells, decodedBinding.atoms))
      if (cell != atom.cell)
        return invalid("adopted Presburger cell is not canonical");
  }
  std::vector<ArtifactRootReference> spatialMappings(
      constraints->view().spatialMappingReferences().begin(),
      constraints->view().spatialMappingReferences().end());
  for (const SystemSearchBindingDomain &binding : decoded->bindings)
    if (std::holds_alternative<::dataflow::RootedGraphLaunchRef>(binding.key))
      for (const SystemSearchAtom &atom : binding.atoms)
        if (const auto *hierarchical =
                std::get_if<SystemHierarchicalGraphBindingDomain>(&atom.domain))
          spatialMappings.insert(
              spatialMappings.end(),
              hierarchical->compatibleSpatialMappings.begin(),
              hierarchical->compatibleSpatialMappings.end());
  auto spatialCatalog =
      detail::importSpatialCatalog(spatialMappings, *dataflow, *system, store);
  if (!spatialCatalog)
    return spatialCatalog.takeError();
  if (llvm::Error error = detail::validateSystemBindingDomains(
          *dataflow, *system, decoded->bindings, *constraintIndex,
          *spatialCatalog))
    return std::move(error);
  if (llvm::Error error = detail::validateSystemServiceDomains(
          *dataflow, *system, decoded->roots, decoded->bindings,
          decoded->services, *constraintIndex, *spatialCatalog))
    return std::move(error);
  auto encoded =
      encodeView(decoded->dataflowReference, decoded->fabricReference,
                 decoded->constraintReference, decoded->roots,
                 decoded->bindings, decoded->services);
  if (!encoded)
    return encoded.takeError();
  if (llvm::ArrayRef(*encoded) != canonicalViewBytes)
    return invalid("adopted view does not re-encode byte-exactly");
  return detail::SystemPnrSearchDomainViewBuilder::create(
      std::move(decoded->dataflowReference),
      std::move(decoded->fabricReference),
      std::move(decoded->constraintReference), std::move(decoded->roots),
      std::move(decoded->bindings), std::move(decoded->services),
      std::move(*encoded), digest);
}

} // namespace loom::pnr
