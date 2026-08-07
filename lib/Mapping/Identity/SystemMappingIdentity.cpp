#include "Mapping/Artifact/SystemMappingIdentity.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>

namespace loom::mapping {
namespace {

using Bytes = std::vector<std::uint8_t>;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_mapping_identity_invalid: " + message);
}

void appendU32(Bytes &bytes, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendU64(Bytes &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendSized(Bytes &bytes, llvm::ArrayRef<std::uint8_t> value) {
  appendU64(bytes, value.size());
  bytes.insert(bytes.end(), value.begin(), value.end());
}

class Reader final {
public:
  explicit Reader(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<std::uint32_t> u32() {
    if (bytes_.size() < sizeof(std::uint32_t))
      return invalid("truncated variant discriminant");
    std::uint32_t value = 0;
    for (unsigned index = 0; index < sizeof(std::uint32_t); ++index)
      value = (value << 8) | bytes_[index];
    bytes_ = bytes_.drop_front(sizeof(std::uint32_t));
    return value;
  }

  llvm::Expected<std::uint64_t> u64() {
    if (bytes_.size() < sizeof(std::uint64_t))
      return invalid("truncated ordinal or size");
    std::uint64_t value = 0;
    for (unsigned index = 0; index < sizeof(std::uint64_t); ++index)
      value = (value << 8) | bytes_[index];
    bytes_ = bytes_.drop_front(sizeof(std::uint64_t));
    return value;
  }

  llvm::Expected<llvm::ArrayRef<std::uint8_t>> sized() {
    auto size = u64();
    if (!size)
      return size.takeError();
    if (*size > bytes_.size() ||
        *size >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
      return invalid("framed reference exceeds remaining bytes");
    llvm::ArrayRef<std::uint8_t> result =
        bytes_.take_front(static_cast<std::size_t>(*size));
    bytes_ = bytes_.drop_front(static_cast<std::size_t>(*size));
    return result;
  }

  bool empty() const { return bytes_.empty(); }
  std::size_t remainingSize() const { return bytes_.size(); }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
};

template <typename Ref>
llvm::Expected<Bytes> dataflowBytes(const ArtifactIdentity &owner,
                                    const Ref &reference) {
  return ::dataflow::encodeDataflowReference(owner, reference);
}

template <typename Ref>
llvm::Expected<Ref> decodeDataflowBytes(Reader &reader,
                                        const ArtifactIdentity &owner) {
  auto bytes = reader.sized();
  if (!bytes)
    return bytes.takeError();
  auto decoded = ::dataflow::decodeDataflowReference<Ref>(*bytes, owner);
  if (!decoded)
    return decoded.takeError();
  auto canonical = ::dataflow::encodeDataflowReference(owner, *decoded);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef(*canonical) != *bytes)
    return invalid("nested Dataflow reference is not canonical");
  return *decoded;
}

std::string stringKey(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

template <typename Ref>
llvm::Error canonicalizeDataflowRefs(std::vector<Ref> &values,
                                     const ArtifactIdentity &owner) {
  std::vector<std::pair<Bytes, Ref>> keyed;
  keyed.reserve(values.size());
  for (const Ref &value : values) {
    auto bytes = dataflowBytes(owner, value);
    if (!bytes)
      return bytes.takeError();
    keyed.emplace_back(std::move(*bytes), value);
  }
  llvm::sort(keyed, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  values.clear();
  for (auto &entry : keyed)
    if (values.empty() || entry.second != values.back())
      values.push_back(std::move(entry.second));
  return llvm::Error::success();
}

struct MutableObligation final {
  SystemServiceObligationProjection projection;
};

llvm::Expected<ServicePlanSelectionAnchor>
decodeServicePlanSelectionAnchorPrefix(Reader &reader,
                                       const ArtifactIdentity &owner) {
  auto kind = reader.u32();
  if (!kind)
    return kind.takeError();
  if (*kind == 0) {
    auto member =
        decodeDataflowBytes<::dataflow::ServiceMemberRef>(reader, owner);
    if (!member)
      return member.takeError();
    return ServicePlanSelectionAnchor{
        ServiceMemberPlanSelectionAnchor{std::move(*member)}};
  }
  if (*kind == 1) {
    auto exposure =
        decodeDataflowBytes<::dataflow::MemoryExposureRef>(reader, owner);
    if (!exposure)
      return exposure.takeError();
    return ServicePlanSelectionAnchor{
        MemoryExposurePlanSelectionAnchor{std::move(*exposure)}};
  }
  return invalid("unknown service-plan selection-anchor kind");
}

} // namespace

llvm::Expected<Bytes>
encodeSystemServiceObligationKey(const ArtifactIdentity &dataflowIdentity,
                                 const SystemServiceObligationKey &key) {
  Bytes result;
  if (const auto *transfer = std::get_if<TransferObligationFamilyKey>(&key)) {
    appendU32(result, 0);
    auto reference = dataflowBytes(dataflowIdentity, *transfer);
    if (!reference)
      return reference.takeError();
    appendSized(result, *reference);
    return result;
  }
  const auto &operation = std::get<OperationServiceObligationFamilyKey>(key);
  if (const auto *memory =
          std::get_if<::dataflow::LogicalMemoryRootOrViewRef>(&operation)) {
    appendU32(result, 1);
    auto reference = dataflowBytes(dataflowIdentity, *memory);
    if (!reference)
      return reference.takeError();
    appendSized(result, *reference);
    return result;
  }
  appendU32(result, 2);
  auto reference = dataflowBytes(
      dataflowIdentity, std::get<::dataflow::FenceActorFamilyRef>(operation));
  if (!reference)
    return reference.takeError();
  appendSized(result, *reference);
  return result;
}

llvm::Expected<SystemServiceObligationKey>
decodeSystemServiceObligationKey(llvm::ArrayRef<std::uint8_t> bytes,
                                 const ArtifactIdentity &dataflowIdentity) {
  Reader reader(bytes);
  auto kind = reader.u32();
  if (!kind)
    return kind.takeError();
  std::optional<SystemServiceObligationKey> result;
  if (*kind == 0) {
    auto reference =
        decodeDataflowBytes<::dataflow::CanonicalProducerTerminalRef>(
            reader, dataflowIdentity);
    if (!reference)
      return reference.takeError();
    result = SystemServiceObligationKey{std::move(*reference)};
  } else if (*kind == 1) {
    auto reference =
        decodeDataflowBytes<::dataflow::LogicalMemoryRootOrViewRef>(
            reader, dataflowIdentity);
    if (!reference)
      return reference.takeError();
    result = SystemServiceObligationKey{
        OperationServiceObligationFamilyKey{std::move(*reference)}};
  } else if (*kind == 2) {
    auto reference = decodeDataflowBytes<::dataflow::FenceActorFamilyRef>(
        reader, dataflowIdentity);
    if (!reference)
      return reference.takeError();
    result = SystemServiceObligationKey{
        OperationServiceObligationFamilyKey{std::move(*reference)}};
  } else {
    return invalid("unknown service-obligation kind");
  }
  if (!reader.empty())
    return invalid("trailing service-obligation bytes");
  auto canonical = encodeSystemServiceObligationKey(dataflowIdentity, *result);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef(*canonical) != bytes)
    return invalid("service-obligation key is not canonical");
  return std::move(*result);
}

llvm::Expected<Bytes>
encodeExecutionContextKey(const ExecutionContextKey &key) {
  Bytes result;
  if (const auto *instruction =
          std::get_if<InstructionExecutionContextKey>(&key)) {
    appendU32(result, 0);
    appendSized(result,
                ::loom::fabric::canonicalFabricBytes(instruction->accCore));
    return result;
  }
  const auto &spatial = std::get<SpatialExecutionContextKey>(key);
  appendU32(result, 1);
  appendSized(result, ::loom::fabric::canonicalFabricBytes(spatial.accCore));
  appendSized(result, spatial.spatialMapping.bytes());
  return result;
}

llvm::Expected<ExecutionContextKey>
decodeExecutionContextKey(llvm::ArrayRef<std::uint8_t> bytes) {
  Reader reader(bytes);
  auto kind = reader.u32();
  if (!kind)
    return kind.takeError();
  auto coreBytes = reader.sized();
  if (!coreBytes)
    return coreBytes.takeError();
  auto core =
      ::loom::fabric::decodeFabricRef<::loom::fabric::AccCoreOccurrenceRef>(
          *coreBytes);
  if (!core)
    return core.takeError();
  const Bytes canonicalCore = ::loom::fabric::canonicalFabricBytes(*core);
  if (llvm::ArrayRef(canonicalCore) != *coreBytes)
    return invalid("execution-context AccCore reference is not canonical");

  std::optional<ExecutionContextKey> result;
  if (*kind == 0) {
    result = InstructionExecutionContextKey{std::move(*core)};
  } else if (*kind == 1) {
    auto identityBytes = reader.sized();
    if (!identityBytes)
      return identityBytes.takeError();
    auto spatialMapping = ArtifactIdentity::fromBytes(*identityBytes);
    if (!spatialMapping)
      return spatialMapping.takeError();
    result = SpatialExecutionContextKey{std::move(*core),
                                        std::move(*spatialMapping)};
  } else {
    return invalid("unknown execution-context kind");
  }
  if (!reader.empty())
    return invalid("trailing execution-context bytes");
  auto canonical = encodeExecutionContextKey(*result);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef(*canonical) != bytes)
    return invalid("execution-context key is not canonical");
  return std::move(*result);
}

llvm::Expected<Bytes>
encodeServicePlanSelectionAnchor(const ArtifactIdentity &dataflowIdentity,
                                 const ServicePlanSelectionAnchor &anchor) {
  Bytes result;
  if (const auto *member =
          std::get_if<ServiceMemberPlanSelectionAnchor>(&anchor)) {
    appendU32(result, 0);
    auto payload = dataflowBytes(dataflowIdentity, member->member);
    if (!payload)
      return payload.takeError();
    appendSized(result, *payload);
    return result;
  }
  appendU32(result, 1);
  auto payload = dataflowBytes(
      dataflowIdentity,
      std::get<MemoryExposurePlanSelectionAnchor>(anchor).exposure);
  if (!payload)
    return payload.takeError();
  appendSized(result, *payload);
  return result;
}

llvm::Expected<ServicePlanSelectionAnchor>
decodeServicePlanSelectionAnchor(llvm::ArrayRef<std::uint8_t> bytes,
                                 const ArtifactIdentity &dataflowIdentity) {
  Reader reader(bytes);
  auto result =
      decodeServicePlanSelectionAnchorPrefix(reader, dataflowIdentity);
  if (!result)
    return result.takeError();
  if (!reader.empty())
    return invalid("trailing service-plan selection-anchor bytes");
  auto canonical = encodeServicePlanSelectionAnchor(dataflowIdentity, *result);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef(*canonical) != bytes)
    return invalid("service-plan selection anchor is not canonical");
  return std::move(*result);
}

llvm::Expected<Bytes>
encodeServicePlanSelectionKey(const ArtifactIdentity &dataflowIdentity,
                              const ServicePlanSelectionKey &key) {
  auto anchor = encodeServicePlanSelectionAnchor(dataflowIdentity, key.anchor);
  if (!anchor)
    return anchor.takeError();
  auto context = encodeExecutionContextKey(key.context);
  if (!context)
    return context.takeError();
  Bytes result = std::move(*anchor);
  appendSized(result, *context);
  return result;
}

llvm::Expected<ServicePlanSelectionKey>
decodeServicePlanSelectionKey(llvm::ArrayRef<std::uint8_t> bytes,
                              const ArtifactIdentity &dataflowIdentity) {
  Reader reader(bytes);
  auto anchor =
      decodeServicePlanSelectionAnchorPrefix(reader, dataflowIdentity);
  if (!anchor)
    return anchor.takeError();
  auto contextBytes = reader.sized();
  if (!contextBytes)
    return contextBytes.takeError();
  auto context = decodeExecutionContextKey(*contextBytes);
  if (!context)
    return context.takeError();
  if (!reader.empty())
    return invalid("trailing service-plan selection-key bytes");
  ServicePlanSelectionKey result{std::move(*anchor), std::move(*context)};
  auto canonical = encodeServicePlanSelectionKey(dataflowIdentity, result);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef(*canonical) != bytes)
    return invalid("service-plan selection key is not canonical");
  return result;
}

llvm::Expected<Bytes>
encodeCanonicalServiceLegKey(const ArtifactIdentity &dataflowIdentity,
                             const CanonicalServiceLegKey &key) {
  auto obligation =
      encodeSystemServiceObligationKey(dataflowIdentity, key.obligation);
  if (!obligation)
    return obligation.takeError();
  auto member = dataflowBytes(dataflowIdentity, key.member);
  if (!member)
    return member.takeError();
  Bytes result;
  appendSized(result, *obligation);
  appendSized(result, *member);
  appendU64(result, key.ordinal);
  return result;
}

llvm::Expected<CanonicalServiceLegKey>
decodeCanonicalServiceLegKey(llvm::ArrayRef<std::uint8_t> bytes,
                             const ArtifactIdentity &dataflowIdentity) {
  Reader reader(bytes);
  auto obligationBytes = reader.sized();
  if (!obligationBytes)
    return obligationBytes.takeError();
  auto obligation =
      decodeSystemServiceObligationKey(*obligationBytes, dataflowIdentity);
  if (!obligation)
    return obligation.takeError();
  auto member = decodeDataflowBytes<::dataflow::ServiceMemberRef>(
      reader, dataflowIdentity);
  if (!member)
    return member.takeError();
  auto ordinal = reader.u64();
  if (!ordinal)
    return ordinal.takeError();
  if (!reader.empty())
    return invalid("trailing canonical service-leg bytes");
  CanonicalServiceLegKey result{std::move(*obligation), std::move(*member),
                                *ordinal};
  auto canonical = encodeCanonicalServiceLegKey(dataflowIdentity, result);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef(*canonical) != bytes)
    return invalid("canonical service-leg key is not canonical");
  return result;
}

llvm::Expected<Bytes>
encodeSystemTransferTerminalKey(const ArtifactIdentity &dataflowIdentity,
                                const SystemTransferTerminalKey &key) {
  Bytes result;
  if (const auto *source = std::get_if<SystemTransferSourceTerminalKey>(&key)) {
    appendU32(result, 0);
    auto leg = encodeCanonicalServiceLegKey(dataflowIdentity, source->leg);
    if (!leg)
      return leg.takeError();
    appendSized(result, *leg);
    return result;
  }
  const auto &sink = std::get<SystemTransferSinkTerminalKey>(key);
  appendU32(result, 1);
  auto leg = encodeCanonicalServiceLegKey(dataflowIdentity, sink.leg);
  if (!leg)
    return leg.takeError();
  appendSized(result, *leg);
  appendU64(result, sink.sinkOrdinal);
  return result;
}

llvm::Expected<DecodedSystemTransferTerminalKeyPrefix>
decodeSystemTransferTerminalKeyPrefix(
    llvm::ArrayRef<std::uint8_t> bytes,
    const ArtifactIdentity &dataflowIdentity) {
  Reader reader(bytes);
  auto kind = reader.u32();
  if (!kind)
    return kind.takeError();
  auto legBytes = reader.sized();
  if (!legBytes)
    return legBytes.takeError();
  auto leg = decodeCanonicalServiceLegKey(*legBytes, dataflowIdentity);
  if (!leg)
    return leg.takeError();

  std::optional<SystemTransferTerminalKey> key;
  if (*kind == 0) {
    key = SystemTransferSourceTerminalKey{std::move(*leg)};
  } else if (*kind == 1) {
    auto sinkOrdinal = reader.u64();
    if (!sinkOrdinal)
      return sinkOrdinal.takeError();
    key = SystemTransferSinkTerminalKey{std::move(*leg), *sinkOrdinal};
  } else {
    return invalid("unknown transfer-terminal key kind");
  }

  const std::size_t byteCount = bytes.size() - reader.remainingSize();
  auto canonical = encodeSystemTransferTerminalKey(dataflowIdentity, *key);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef(*canonical) != bytes.take_front(byteCount))
    return invalid("transfer-terminal key is not canonical");
  return DecodedSystemTransferTerminalKeyPrefix{std::move(*key), byteCount};
}

llvm::Expected<SystemTransferTerminalKey>
decodeSystemTransferTerminalKey(llvm::ArrayRef<std::uint8_t> bytes,
                                const ArtifactIdentity &dataflowIdentity) {
  auto decoded = decodeSystemTransferTerminalKeyPrefix(bytes, dataflowIdentity);
  if (!decoded)
    return decoded.takeError();
  if (decoded->byteCount != bytes.size())
    return invalid("trailing transfer-terminal key bytes");
  return std::move(decoded->key);
}

llvm::Expected<std::vector<SystemServiceObligationProjection>>
projectSystemServiceObligations(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> rootThreadLaunches) {
  std::vector<::dataflow::RootThreadLaunchRef> roots(rootThreadLaunches.begin(),
                                                     rootThreadLaunches.end());
  if (llvm::Error error = canonicalizeDataflowRefs(roots, dataflow.identity()))
    return std::move(error);
  if (roots.empty())
    return invalid("root thread launch scope is empty");
  std::vector<bool> selectedRootEntities(dataflow.entityCount(), false);
  for (const auto &root : roots)
    if (llvm::Error error = dataflow.resolve(root).takeError())
      return std::move(error);
    else
      selectedRootEntities[root.entity.value()] = true;

  std::vector<MutableObligation> obligations;
  std::map<std::string, std::size_t> obligationIndex;
  auto get = [&](SystemServiceObligationKey key)
      -> llvm::Expected<MutableObligation *> {
    auto bytes = encodeSystemServiceObligationKey(dataflow.identity(), key);
    if (!bytes)
      return bytes.takeError();
    auto [it, inserted] =
        obligationIndex.emplace(stringKey(*bytes), obligations.size());
    if (inserted)
      obligations.push_back(
          {SystemServiceObligationProjection{std::move(key), {}, {}, {}, {}}});
    return &obligations[it->second];
  };

  for (const auto &root : roots) {
    if (llvm::Error error = dataflow.forEachProducerTerminal(
            root,
            [&](const ::dataflow::CanonicalProducerTerminalView &view)
                -> llvm::Error {
              auto obligation = get(SystemServiceObligationKey{view.terminal});
              if (!obligation)
                return obligation.takeError();
              auto member = dataflow.messageTransferMember(view.terminal);
              if (!member)
                return member.takeError();
              auto service =
                  ::dataflow::semantics::CanonicalService::messageTransfer(
                      view.payloadType);
              if (!service)
                return service.takeError();
              (*obligation)->projection.members.push_back(*member);
              if (llvm::Error sinkError = dataflow.pairedSinks(
                      view.terminal,
                      [&](const ::dataflow::CanonicalSinkTerminalRef &sink) {
                        (*obligation)->projection.sinks.push_back(sink);
                      }))
                return sinkError;
              for (unsigned ordinal = 0; ordinal < service->legCount();
                   ++ordinal)
                (*obligation)
                    ->projection.legs.push_back(
                        {(*obligation)->projection.key, *member, ordinal});
              return llvm::Error::success();
            }))
      return std::move(error);

    if (llvm::Error error = dataflow.forEachContextualServiceActor(
            root,
            [&](::dataflow::ContextualActorRef contextual) -> llvm::Error {
              auto actor = dataflow.resolve(contextual.actor);
              if (!actor)
                return actor.takeError();
              auto member = dataflow.serviceMemberFor(contextual);
              if (!member)
                return member.takeError();
              auto service =
                  ::dataflow::semantics::CanonicalService::forActor(actor->op);
              if (!service)
                return service.takeError();

              std::optional<OperationServiceObligationFamilyKey> operation;
              if (std::holds_alternative<::dataflow::FenceActorMemberRef>(
                      *member)) {
                auto fence = dataflow.asFenceFamily(contextual.actor);
                if (!fence)
                  return fence.takeError();
                operation = OperationServiceObligationFamilyKey{*fence};
              } else {
                auto memory = dataflow.resolveAddressedMemory(contextual);
                if (!memory)
                  return memory.takeError();
                operation = OperationServiceObligationFamilyKey{*memory};
              }
              auto obligation =
                  get(SystemServiceObligationKey{std::move(*operation)});
              if (!obligation)
                return obligation.takeError();
              (*obligation)->projection.members.push_back(*member);
              for (unsigned ordinal = 0; ordinal < service->legCount();
                   ++ordinal)
                (*obligation)
                    ->projection.legs.push_back(
                        {(*obligation)->projection.key, *member, ordinal});
              return llvm::Error::success();
            }))
      return std::move(error);
  }

  llvm::Error exposureError = llvm::Error::success();
  dataflow.forEachMemoryExposure([&](::dataflow::MemoryExposureRef exposure) {
    if (exposureError ||
        !selectedRootEntities[exposure.launch.rootThreadLaunch.entity.value()])
      return;
    auto memory = dataflow.resolveExposure(exposure);
    if (!memory) {
      exposureError = memory.takeError();
      return;
    }
    auto obligation = get(SystemServiceObligationKey{
        OperationServiceObligationFamilyKey{*memory}});
    if (!obligation) {
      exposureError = obligation.takeError();
      return;
    }
    (*obligation)->projection.exposures.push_back(exposure);
  });
  if (exposureError)
    return exposureError;

  std::vector<SystemServiceObligationProjection> result;
  result.reserve(obligations.size());
  for (MutableObligation &entry : obligations) {
    auto &projection = entry.projection;
    if (llvm::Error error =
            canonicalizeDataflowRefs(projection.members, dataflow.identity()))
      return std::move(error);
    if (llvm::Error error =
            canonicalizeDataflowRefs(projection.sinks, dataflow.identity()))
      return std::move(error);
    if (llvm::Error error =
            canonicalizeDataflowRefs(projection.exposures, dataflow.identity()))
      return std::move(error);
    std::vector<std::pair<Bytes, CanonicalServiceLegKey>> legs;
    legs.reserve(projection.legs.size());
    for (const CanonicalServiceLegKey &leg : projection.legs) {
      auto bytes = encodeCanonicalServiceLegKey(dataflow.identity(), leg);
      if (!bytes)
        return bytes.takeError();
      legs.emplace_back(std::move(*bytes), leg);
    }
    llvm::sort(legs, [](const auto &lhs, const auto &rhs) {
      return lhs.first < rhs.first;
    });
    projection.legs.clear();
    for (auto &leg : legs)
      if (projection.legs.empty() || leg.second != projection.legs.back())
        projection.legs.push_back(std::move(leg.second));
    result.push_back(std::move(projection));
  }
  std::vector<std::pair<Bytes, SystemServiceObligationProjection>> ordered;
  ordered.reserve(result.size());
  for (SystemServiceObligationProjection &projection : result) {
    auto bytes =
        encodeSystemServiceObligationKey(dataflow.identity(), projection.key);
    if (!bytes)
      return bytes.takeError();
    ordered.emplace_back(std::move(*bytes), std::move(projection));
  }
  llvm::sort(ordered, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  result.clear();
  for (auto &entry : ordered)
    result.push_back(std::move(entry.second));
  return result;
}

} // namespace loom::mapping
