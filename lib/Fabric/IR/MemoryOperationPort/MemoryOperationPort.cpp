#include "Fabric/IR/MemoryOperationPort.h"

#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Fabric/IR/MemoryCapabilityRelation.h"
#include "MemoryOperationPortInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <limits>
#include <set>
#include <system_error>

using namespace dataflow;
using namespace dataflow::semantics;
using namespace loom::fabric;

namespace fabric {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(std::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

llvm::Expected<std::vector<std::uint8_t>> roleBytes(ServiceValueRole role) {
  auto encoded = dataflow::encodeServiceValueRole(role);
  if (!encoded)
    return encoded.takeError();
  return std::vector<std::uint8_t>(encoded->bytes().begin(),
                                   encoded->bytes().end());
}

llvm::Expected<std::vector<std::uint8_t>>
bindingBytes(llvm::ArrayRef<MemoryRoleEndpointBindingRecord> bindings) {
  std::vector<std::uint8_t> bytes;
  auto appendU64 = [&](std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8)
      bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  };
  appendU64(bindings.size());
  for (const MemoryRoleEndpointBindingRecord &binding : bindings) {
    auto encodedRole = roleBytes(binding.role);
    if (!encodedRole)
      return encodedRole.takeError();
    appendU64(encodedRole->size());
    bytes.insert(bytes.end(), encodedRole->begin(), encodedRole->end());
    appendU64(binding.endpointOrdinal);
  }
  return bytes;
}

class BindingReader {
public:
  explicit BindingReader(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<std::uint64_t> u64(const llvm::Twine &field) {
    if (remaining() < 8)
      return invalid(field + " is truncated");
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }

  llvm::Expected<llvm::ArrayRef<std::uint8_t>> frame(const llvm::Twine &field) {
    auto size = u64(field + " length");
    if (!size)
      return size.takeError();
    if (*size > remaining())
      return invalid(field + " is truncated");
    llvm::ArrayRef<std::uint8_t> value = bytes_.slice(offset_, *size);
    offset_ += *size;
    return value;
  }

  std::size_t remaining() const { return bytes_.size() - offset_; }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

llvm::Expected<std::vector<MemoryRoleEndpointBindingRecord>>
decodeBindingBytes(llvm::ArrayRef<std::uint8_t> bytes) {
  BindingReader reader(bytes);
  auto count = reader.u64("role binding count");
  if (!count || *count > reader.remaining() / 16)
    return count ? invalid("role binding count exceeds its framing")
                 : count.takeError();
  std::vector<MemoryRoleEndpointBindingRecord> bindings;
  bindings.reserve(*count);
  for (std::uint64_t index = 0; index < *count; ++index) {
    auto roleWire = reader.frame("service role");
    if (!roleWire)
      return roleWire.takeError();
    auto role = dataflow::decodeServiceValueRole(*roleWire);
    if (!role)
      return role.takeError();
    auto endpoint = reader.u64("role endpoint ordinal");
    if (!endpoint)
      return endpoint.takeError();
    bindings.push_back({*role, *endpoint});
  }
  if (reader.remaining() != 0)
    return invalid("role binding record has trailing bytes");
  auto canonical = bindingBytes(bindings);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*canonical) != bytes)
    return invalid("role binding record is not canonical");
  return bindings;
}

llvm::Expected<std::vector<MemoryRoleEndpointBindingRecord>>
normalizeBindings(llvm::ArrayRef<MemoryRoleEndpointBindingRecord> bindings) {
  struct EncodedBinding {
    std::vector<std::uint8_t> role;
    MemoryRoleEndpointBindingRecord binding;
  };
  std::vector<EncodedBinding> ordered;
  ordered.reserve(bindings.size());
  for (MemoryRoleEndpointBindingRecord binding : bindings) {
    auto encoded = roleBytes(binding.role);
    if (!encoded)
      return encoded.takeError();
    ordered.push_back({std::move(*encoded), binding});
  }
  llvm::sort(ordered, [](const auto &left, const auto &right) {
    return left.role < right.role;
  });
  for (std::size_t index = 1; index < ordered.size(); ++index)
    if (ordered[index - 1].role == ordered[index].role)
      return invalid("memory role binding repeats one service role");
  std::vector<MemoryRoleEndpointBindingRecord> result;
  result.reserve(ordered.size());
  for (const EncodedBinding &entry : ordered)
    result.push_back(entry.binding);
  return result;
}

bool isArgumentRole(ServiceKind kind, ServiceValueRole role) {
  return llvm::is_contained(getServiceRoleSchema(kind).arguments, role);
}

bool isResultRole(ServiceKind kind, ServiceValueRole role) {
  return llvm::is_contained(getServiceRoleSchema(kind).results, role);
}

bool hasDynamicMask(const ParameterizedMemoryAccessDomain &domain) {
  for (const MemoryAccessClass &access : domain.accessClasses())
    for (MaskInactivePair pair : access.maskInactivePairs())
      if (pair.mask == MemoryMaskForm::Dynamic)
        return true;
  return false;
}

llvm::Expected<std::pair<std::uint64_t, std::uint64_t>>
domainBounds(const UnsignedDomain &domain) {
  if (domain.intervals().empty())
    return invalid("unsigned domain has no interval");
  return std::make_pair(domain.intervals().front().lower,
                        domain.intervals().back().upper);
}

llvm::Expected<std::pair<std::uint64_t, std::uint64_t>>
dataWidthBounds(const MemoryAccessClass &access) {
  auto elements = domainBounds(access.elementWidths());
  auto lanes = domainBounds(access.flattenedLaneCounts());
  if (!elements)
    return elements.takeError();
  if (!lanes)
    return lanes.takeError();
  auto minimum = llvm::checkedMulUnsigned(elements->first, lanes->first);
  if (!minimum)
    return invalid("memory access payload width overflows u64");
  auto envelope = deriveMemoryAccessTransportEnvelope(access);
  if (!envelope)
    return envelope.takeError();
  return std::make_pair(*minimum, envelope->dataPayloadBits);
}

bool everyElementWidthIsByteMultiple(const UnsignedDomain &domain) {
  for (UnsignedInterval interval : domain.intervals())
    if (interval.lower != interval.upper || interval.lower % 8 != 0)
      return false;
  return true;
}

enum class PatternRequirement { Direct, ActiveLanes, Either };

PatternRequirement requirementForPlain(const MemoryAccessClass &access) {
  return access.accessForm() == MemoryAccessForm::Element
             ? PatternRequirement::Direct
             : PatternRequirement::Either;
}

PatternRequirement
requirementForGranularity(std::optional<VectorAtomicGranularity> granularity) {
  return granularity == VectorAtomicGranularity::PerLane
             ? PatternRequirement::ActiveLanes
             : PatternRequirement::Direct;
}

bool projectionSatisfies(MemoryPortTransactionProjection projection,
                         PatternRequirement requirement) {
  if (requirement == PatternRequirement::Either)
    return true;
  return (projection == MemoryPortTransactionProjection::Direct) ==
         (requirement == PatternRequirement::Direct);
}

template <typename Callback>
llvm::Error
forEachPatternRequirement(const MemoryActorContractDomain &actors,
                          const ParameterizedMemoryAccessDomain *accesses,
                          Callback callback) {
  for (const MemoryActorContractClause &clause : actors.clauses()) {
    if (!accesses) {
      if (!std::holds_alternative<FenceContractClause>(clause))
        return invalid("addressed actor contract has no access domain");
      if (llvm::Error error = callback(PatternRequirement::Direct, nullptr))
        return error;
      continue;
    }
    for (const MemoryAccessClass &access : accesses->accessClasses()) {
      if (std::holds_alternative<LoadStorePlainContractClause>(clause)) {
        if (llvm::Error error = callback(requirementForPlain(access), &access))
          return error;
        continue;
      }
      llvm::Error error = std::visit(
          [&](const auto &typed) -> llvm::Error {
            using Clause = std::decay_t<decltype(typed)>;
            if constexpr (std::is_same_v<Clause,
                                         LoadStorePlainContractClause> ||
                          std::is_same_v<Clause, FenceContractClause>) {
              return llvm::Error::success();
            } else {
              for (auto granularity : typed.vectorGranularityValues)
                if (llvm::Error nested = callback(
                        requirementForGranularity(granularity), &access))
                  return nested;
              return llvm::Error::success();
            }
          },
          clause);
      if (error)
        return error;
    }
  }
  return llvm::Error::success();
}

const MemoryTransportEndpointDescriptor *
endpointAt(llvm::ArrayRef<MemoryTransportEndpointDescriptor> endpoints,
           std::uint64_t ordinal) {
  return ordinal < endpoints.size() ? &endpoints[ordinal] : nullptr;
}

llvm::Expected<const MemoryRoleEndpointBindingRecord *>
bindingFor(llvm::ArrayRef<MemoryRoleEndpointBindingRecord> bindings,
           ServiceValueRole role) {
  for (const MemoryRoleEndpointBindingRecord &binding : bindings)
    if (binding.role == role)
      return &binding;
  return invalid("memory capability omits a required role binding");
}

llvm::Error validateSubwordRelation(
    OperationSchemaId schema, const MemoryAccessClass &access,
    llvm::ArrayRef<MemoryRoleEndpointBindingRecord> bindings,
    llvm::ArrayRef<MemoryTransportEndpointDescriptor> endpoints) {
  auto widths = dataWidthBounds(access);
  if (!widths)
    return widths.takeError();

  const bool reads = schema == OperationSchemaId::DataflowLoad ||
                     schema == OperationSchemaId::DataflowAtomicRmw ||
                     schema == OperationSchemaId::DataflowCmpXchg;
  const bool writes = schema == OperationSchemaId::DataflowStore ||
                      schema == OperationSchemaId::DataflowAtomicRmw ||
                      schema == OperationSchemaId::DataflowCmpXchg;

  auto validateReadRole = [&](ServiceValueRole role) -> llvm::Error {
    auto binding = bindingFor(bindings, role);
    if (!binding)
      return binding.takeError();
    const auto *endpoint = endpointAt(endpoints, (*binding)->endpointOrdinal);
    if (!endpoint)
      return invalid("memory result role references an unknown endpoint");
    for (ReadSubwordSemantics semantics :
         access.readSubwordSemantics().values()) {
      if (semantics == ReadSubwordSemantics::NotApplicable)
        return invalid(
            "read-like capability uses NotApplicable read semantics");
      if (semantics == ReadSubwordSemantics::Exact &&
          (widths->first != widths->second ||
           widths->first != endpoint->payloadWidth))
        return invalid("Exact read semantics disagree with endpoint width");
      if (semantics == ReadSubwordSemantics::ZeroExtend &&
          widths->second > endpoint->payloadWidth)
        return invalid(
            "ZeroExtend read payload width " + llvm::Twine(widths->second) +
            " exceeds endpoint width " + llvm::Twine(endpoint->payloadWidth));
    }
    return llvm::Error::success();
  };

  auto validateWriteRole = [&](ServiceValueRole role) -> llvm::Error {
    auto binding = bindingFor(bindings, role);
    if (!binding)
      return binding.takeError();
    const auto *endpoint = endpointAt(endpoints, (*binding)->endpointOrdinal);
    if (!endpoint)
      return invalid("memory input role references an unknown endpoint");
    for (WriteSubwordSemantics semantics :
         access.writeSubwordSemantics().values()) {
      if (semantics == WriteSubwordSemantics::NotApplicable)
        return invalid(
            "write-like capability uses NotApplicable write semantics");
      if (semantics == WriteSubwordSemantics::Exact &&
          (widths->first != widths->second ||
           widths->first != endpoint->payloadWidth))
        return invalid("Exact write semantics disagree with endpoint width");
      if (semantics == WriteSubwordSemantics::ByteEnable &&
          (widths->second > endpoint->payloadWidth ||
           !everyElementWidthIsByteMultiple(access.elementWidths())))
        return invalid("ByteEnable write payload is not a fitting byte range");
    }
    return llvm::Error::success();
  };

  if (!reads && (access.readSubwordSemantics().values().size() != 1 ||
                 access.readSubwordSemantics().values().front() !=
                     ReadSubwordSemantics::NotApplicable))
    return invalid("write-only capability has read subword semantics");
  if (!writes && (access.writeSubwordSemantics().values().size() != 1 ||
                  access.writeSubwordSemantics().values().front() !=
                      WriteSubwordSemantics::NotApplicable))
    return invalid("read-only capability has write subword semantics");

  if (reads) {
    ServiceValueRole role = schema == OperationSchemaId::DataflowLoad
                                ? ServiceValueRole::Data
                                : ServiceValueRole::Old;
    if (llvm::Error error = validateReadRole(role))
      return error;
  }
  if (writes) {
    ServiceValueRole role = ServiceValueRole::Data;
    if (schema == OperationSchemaId::DataflowAtomicRmw)
      role = ServiceValueRole::Update;
    else if (schema == OperationSchemaId::DataflowCmpXchg)
      role = ServiceValueRole::Expected;
    if (llvm::Error error = validateWriteRole(role))
      return error;
    if (schema == OperationSchemaId::DataflowCmpXchg)
      if (llvm::Error error = validateWriteRole(ServiceValueRole::Desired))
        return error;
  }
  return llvm::Error::success();
}

llvm::Error validateRoleRelation(
    Schedule schedule,
    llvm::ArrayRef<MemoryTransportEndpointDescriptor> endpoints,
    llvm::ArrayRef<std::uint64_t> inventory,
    const MemoryCapabilityAlternativeRecord &alternative) {
  auto kind =
      getMemoryServiceKind(alternative.actorContractDomain.actorSchema());
  if (!kind)
    return kind.takeError();
  const bool fence = *kind == ServiceKind::MemoryFence;
  if (fence != !alternative.accessDomain)
    return invalid("memory access domain is absent exactly for fence");

  std::vector<ServiceValueRole> expected;
  const ServiceRoleSchema &schema = getServiceRoleSchema(*kind);
  const bool includeMask =
      alternative.accessDomain && hasDynamicMask(*alternative.accessDomain);
  for (ServiceValueRole role : schema.arguments)
    if (role != ServiceValueRole::Mask || includeMask)
      expected.push_back(role);
  expected.insert(expected.end(), schema.results.begin(), schema.results.end());
  if (alternative.roleToEndpoint.size() != expected.size())
    return invalid("memory role-to-endpoint relation is not total");
  for (ServiceValueRole role : expected) {
    const bool present =
        llvm::any_of(alternative.roleToEndpoint, [role](const auto &binding) {
          return binding.role == role;
        });
    if (!present)
      return invalid("memory role-to-endpoint relation omits a maximal role");
  }

  std::set<std::uint64_t> activeInputs;
  std::set<std::uint64_t> activeOutputs;
  for (const MemoryRoleEndpointBindingRecord &binding :
       alternative.roleToEndpoint) {
    if (!llvm::is_contained(inventory, binding.endpointOrdinal))
      return invalid("memory role binding is outside the port inventory");
    const auto *endpoint = endpointAt(endpoints, binding.endpointOrdinal);
    if (!endpoint)
      return invalid("memory role binding is outside the occurrence signature");
    const bool input = isArgumentRole(*kind, binding.role);
    if (!input && !isResultRole(*kind, binding.role))
      return invalid("memory role binding is not in its service schema");
    const FabricPortDirection expectedDirection =
        input ? FabricPortDirection::Input : FabricPortDirection::Output;
    if (endpoint->direction != expectedDirection)
      return invalid("memory role binding has the wrong endpoint direction");
    if ((schedule == Schedule::Spatial) != !endpoint->tagWidth)
      return invalid("memory endpoint kind does not match engine schedule");
    if (input) {
      if (schedule == Schedule::Spatial &&
          !activeInputs.insert(binding.endpointOrdinal).second)
        return invalid("Spatial memory input role bindings are not injective");
    } else if (!activeOutputs.insert(binding.endpointOrdinal).second) {
      return invalid("memory output role bindings are not injective");
    }
  }

  if (!alternative.accessDomain)
    return llvm::Error::success();
  for (const MemoryAccessClass &access :
       alternative.accessDomain->accessClasses()) {
    if (llvm::Error error = validateSubwordRelation(
            alternative.actorContractDomain.actorSchema(), access,
            alternative.roleToEndpoint, endpoints))
      return error;
    auto addressBinding =
        bindingFor(alternative.roleToEndpoint, ServiceValueRole::Address);
    if (!addressBinding)
      return addressBinding.takeError();
    const auto *address =
        endpointAt(endpoints, (*addressBinding)->endpointOrdinal);
    if (!address || address->payloadWidth == 0)
      return invalid("memory address endpoint has zero payload capacity");
    auto envelope = deriveMemoryAccessTransportEnvelope(access);
    if (!envelope)
      return envelope.takeError();
    if (envelope->addressPayloadBits > address->payloadWidth)
      return invalid("memory address payload exceeds endpoint width");
    if (envelope->maskPayloadBits != 0) {
      auto maskBinding =
          bindingFor(alternative.roleToEndpoint, ServiceValueRole::Mask);
      if (!maskBinding)
        return maskBinding.takeError();
      const auto *mask = endpointAt(endpoints, (*maskBinding)->endpointOrdinal);
      if (!mask || mask->payloadWidth < envelope->maskPayloadBits)
        return invalid("memory mask payload exceeds endpoint width");
    }
    if (*kind == ServiceKind::MemoryCompareExchange) {
      auto successBinding =
          bindingFor(alternative.roleToEndpoint, ServiceValueRole::Success);
      if (!successBinding)
        return successBinding.takeError();
      const auto *success =
          endpointAt(endpoints, (*successBinding)->endpointOrdinal);
      auto lanes = domainBounds(access.flattenedLaneCounts());
      if (!lanes)
        return lanes.takeError();
      if (!success || success->payloadWidth < lanes->second)
        return invalid(
            "compare-exchange success payload exceeds endpoint width");
    }
  }
  return llvm::Error::success();
}

llvm::Expected<std::vector<MemoryCapabilityAlternativeRecord>>
normalizeAlternatives(mlir::MLIRContext *context,
                      llvm::ArrayRef<MemoryCapabilityAlternativeRecord> input) {
  std::vector<detail::MemoryCapabilityRelationEntry> relation;
  relation.reserve(input.size());
  for (const MemoryCapabilityAlternativeRecord &alternative : input) {
    auto physicalFacts = bindingBytes(alternative.roleToEndpoint);
    if (!physicalFacts)
      return physicalFacts.takeError();
    relation.push_back({alternative.actorContractDomain,
                        alternative.accessDomain, std::move(*physicalFacts),
                        alternative.admissibleUsePatterns});
  }

  auto normalized =
      detail::normalizeMemoryCapabilityRelation(context, relation);
  if (!normalized)
    return normalized.takeError();

  std::vector<MemoryCapabilityAlternativeRecord> result;
  result.reserve(normalized->size());
  for (detail::MemoryCapabilityRelationEntry &entry : *normalized) {
    auto bindings = decodeBindingBytes(entry.physicalFacts);
    if (!bindings)
      return bindings.takeError();
    result.push_back({std::move(entry.actorContractDomain),
                      std::move(*bindings), std::move(entry.accessDomain),
                      std::move(entry.admissibleUsePatterns)});
  }
  return result;
}
llvm::Error validateAlternatives(
    Schedule schedule,
    llvm::ArrayRef<MemoryTransportEndpointDescriptor> endpoints,
    llvm::ArrayRef<std::uint64_t> inventory, const ResourceContract &contract,
    llvm::ArrayRef<MemoryOperationPatternRecord> operationPatterns,
    llvm::ArrayRef<MemoryCapabilityAlternativeRecord> alternatives) {
  std::vector<std::optional<std::uint64_t>> activeLaneMaximum(
      contract.usePatternCount());
  std::vector<bool> patternUsed(contract.usePatternCount(), false);
  std::vector<bool> endpointUsed(endpoints.size(), false);

  for (const MemoryCapabilityAlternativeRecord &alternative : alternatives) {
    if (llvm::Error error =
            validateRoleRelation(schedule, endpoints, inventory, alternative))
      return error;
    for (const MemoryRoleEndpointBindingRecord &binding :
         alternative.roleToEndpoint)
      endpointUsed[binding.endpointOrdinal] = true;

    for (UsePatternKey pattern : alternative.admissibleUsePatterns) {
      if (pattern.ordinal() >= contract.usePatternCount())
        return invalid("memory capability references an unknown use pattern");
      patternUsed[pattern.ordinal()] = true;
    }

    if (llvm::Error error = forEachPatternRequirement(
            alternative.actorContractDomain,
            alternative.accessDomain ? &*alternative.accessDomain : nullptr,
            [&](PatternRequirement requirement,
                const MemoryAccessClass *access) -> llvm::Error {
              for (UsePatternKey pattern : alternative.admissibleUsePatterns) {
                const MemoryPortTransactionProjection projection =
                    operationPatterns[pattern.ordinal()].transactionProjection;
                if (!projectionSatisfies(projection, requirement))
                  return invalid(
                      "memory use pattern is illegal for an admitted tuple");
                if (projection ==
                    MemoryPortTransactionProjection::ActiveLanesRowMajor) {
                  if (!access ||
                      access->accessForm() == MemoryAccessForm::Element)
                    return invalid(
                        "ActiveLanesRowMajor requires vector access");
                  auto lanes = domainBounds(access->flattenedLaneCounts());
                  if (!lanes)
                    return lanes.takeError();
                  auto &maximum = activeLaneMaximum[pattern.ordinal()];
                  maximum = std::max(maximum.value_or(0), lanes->second);
                }
              }
              return llvm::Error::success();
            }))
      return error;
  }

  for (std::uint32_t ordinal = 0; ordinal < contract.usePatternCount();
       ++ordinal) {
    if (!patternUsed[ordinal])
      return invalid("memory operation port has an unreachable use pattern");
    const UsePattern pattern = contract.usePattern(UsePatternKey(ordinal));
    const MemoryPortTransactionProjection projection =
        operationPatterns[ordinal].transactionProjection;
    if (projection == MemoryPortTransactionProjection::Direct) {
      if (pattern.internalTransactionCount != 1)
        return invalid("Direct memory pattern must have one transaction slot");
      continue;
    }
    if (!activeLaneMaximum[ordinal])
      return invalid("ActiveLanesRowMajor pattern is unreachable");
    if (pattern.internalTransactionCount != *activeLaneMaximum[ordinal])
      return invalid(
          "ActiveLanesRowMajor transaction count is not its lane maximum");
  }

  for (std::uint64_t endpoint : inventory)
    if (!endpointUsed[endpoint])
      return invalid("memory operation port inventory has an unused endpoint");
  return llvm::Error::success();
}

llvm::Expected<MemoryOperationPortDeclaration> normalizeDeclaration(
    mlir::MLIRContext *context, Schedule schedule,
    llvm::ArrayRef<MemoryTransportEndpointDescriptor> endpoints,
    MemoryOperationPortDeclaration declaration) {
  if (!context)
    return invalid("memory operation port requires an MLIR context");
  if (schedule != Schedule::Spatial && schedule != Schedule::Temporal)
    return invalid("memory operation port has an unknown schedule");
  if (declaration.endpointInventory.empty())
    return invalid("memory operation port has an empty endpoint inventory");
  llvm::sort(declaration.endpointInventory);
  if (std::adjacent_find(declaration.endpointInventory.begin(),
                         declaration.endpointInventory.end()) !=
      declaration.endpointInventory.end())
    return invalid("memory operation port repeats an endpoint ordinal");
  if (declaration.endpointInventory.back() >= endpoints.size())
    return invalid("memory operation port endpoint is outside its owner");
  for (std::uint64_t endpoint : declaration.endpointInventory) {
    const auto &descriptor = endpoints[endpoint];
    if ((schedule == Schedule::Spatial) != !descriptor.tagWidth)
      return invalid("memory port endpoint kind disagrees with its schedule");
  }

  if (declaration.operationPatternSemantics.size() !=
      declaration.resourceContract.usePatternCount())
    return invalid(
        "memory operation pattern table does not match ResourceContract");
  for (MemoryOperationPatternRecord pattern :
       declaration.operationPatternSemantics)
    if (pattern.transactionProjection !=
            MemoryPortTransactionProjection::Direct &&
        pattern.transactionProjection !=
            MemoryPortTransactionProjection::ActiveLanesRowMajor)
      return invalid("memory operation pattern has an unknown projection");

  for (MemoryCapabilityAlternativeRecord &alternative :
       declaration.capabilityAlternatives) {
    auto bindings = normalizeBindings(alternative.roleToEndpoint);
    if (!bindings)
      return bindings.takeError();
    alternative.roleToEndpoint = std::move(*bindings);
    llvm::sort(alternative.admissibleUsePatterns,
               [](UsePatternKey left, UsePatternKey right) {
                 return left.ordinal() < right.ordinal();
               });
    if (std::adjacent_find(alternative.admissibleUsePatterns.begin(),
                           alternative.admissibleUsePatterns.end()) !=
        alternative.admissibleUsePatterns.end())
      return invalid("memory capability repeats a use pattern");
  }
  auto alternatives =
      normalizeAlternatives(context, declaration.capabilityAlternatives);
  if (!alternatives)
    return alternatives.takeError();
  declaration.capabilityAlternatives = std::move(*alternatives);
  if (llvm::Error error = validateAlternatives(
          schedule, endpoints, declaration.endpointInventory,
          declaration.resourceContract, declaration.operationPatternSemantics,
          declaration.capabilityAlternatives))
    return std::move(error);
  return declaration;
}

bool subwordSupportsActual(
    const MemoryAccessClass &accessClass,
    const CanonicalMemoryAccessView &access,
    llvm::ArrayRef<MemoryRoleEndpointBindingRecord> bindings,
    llvm::ArrayRef<MemoryTransportEndpointDescriptor> endpoints) {
  auto endpointWidth =
      [&](ServiceValueRole role) -> std::optional<std::uint32_t> {
    auto binding = bindingFor(bindings, role);
    if (!binding) {
      llvm::consumeError(binding.takeError());
      return std::nullopt;
    }
    const auto *endpoint = endpointAt(endpoints, (*binding)->endpointOrdinal);
    return endpoint ? std::optional<std::uint32_t>(endpoint->payloadWidth)
                    : std::nullopt;
  };

  const bool reads = access.operation() != MemoryAccessOperation::Store;
  const bool writes = access.operation() != MemoryAccessOperation::Load;
  if (reads) {
    ServiceValueRole role = access.operation() == MemoryAccessOperation::Load
                                ? ServiceValueRole::Data
                                : ServiceValueRole::Old;
    auto width = endpointWidth(role);
    if (!width)
      return false;
    bool accepted = false;
    for (ReadSubwordSemantics semantics :
         accessClass.readSubwordSemantics().values())
      accepted |= semantics == ReadSubwordSemantics::Exact
                      ? access.dataBits() == *width
                      : semantics == ReadSubwordSemantics::ZeroExtend &&
                            access.dataBits() <= *width;
    if (!accepted)
      return false;
  }
  if (writes) {
    ServiceValueRole role = ServiceValueRole::Data;
    if (access.operation() == MemoryAccessOperation::AtomicRmw)
      role = ServiceValueRole::Update;
    else if (access.operation() == MemoryAccessOperation::CompareExchange)
      role = ServiceValueRole::Expected;
    auto width = endpointWidth(role);
    if (!width)
      return false;
    bool accepted = false;
    for (WriteSubwordSemantics semantics :
         accessClass.writeSubwordSemantics().values())
      accepted |= semantics == WriteSubwordSemantics::Exact
                      ? access.dataBits() == *width
                      : semantics == WriteSubwordSemantics::ByteEnable &&
                            access.dataBits() <= *width &&
                            access.dataBits() % 8 == 0;
    if (!accepted)
      return false;
  }
  auto addressWidth = endpointWidth(ServiceValueRole::Address);
  if (!addressWidth || access.addressBits() > *addressWidth)
    return false;
  if (access.maskForm() == MemoryMaskForm::Dynamic) {
    auto maskWidth = endpointWidth(ServiceValueRole::Mask);
    if (!maskWidth || access.maskBits() > *maskWidth)
      return false;
  }
  if (access.operation() == MemoryAccessOperation::CompareExchange) {
    auto successWidth = endpointWidth(ServiceValueRole::Success);
    if (!successWidth || access.laneCount() > *successWidth)
      return false;
  }
  return true;
}

PatternRequirement exactRequirement(const CanonicalActorSchemaProjection &actor,
                                    const CanonicalMemoryAccessView *access) {
  if (!access)
    return PatternRequirement::Direct;
  const auto *memory = std::get_if<MemoryContractPayload>(&actor.payload);
  if (!memory)
    return PatternRequirement::Direct;
  return std::visit(
      [&](const auto &projection) {
        using Projection = std::decay_t<decltype(projection)>;
        if constexpr (std::is_same_v<Projection, PlainAccessProjection>)
          return access->form() == MemoryAccessForm::Element
                     ? PatternRequirement::Direct
                     : PatternRequirement::Either;
        else if constexpr (std::is_same_v<Projection, FenceProjection>)
          return PatternRequirement::Direct;
        else if constexpr (std::is_same_v<Projection, AtomicAccessProjection>)
          return requirementForGranularity(projection.vectorGranularity);
        else if constexpr (std::is_same_v<Projection, AtomicRmwProjection>)
          return requirementForGranularity(projection.access.vectorGranularity);
        else
          return requirementForGranularity(projection.vectorGranularity);
      },
      *memory);
}

} // namespace

llvm::Expected<MemoryOperationPortRecord> MemoryOperationPortRecord::create(
    mlir::MLIRContext *context, Schedule schedule,
    llvm::ArrayRef<MemoryTransportEndpointDescriptor> endpoints,
    MemoryOperationPortDeclaration declaration) {
  auto normalized = normalizeDeclaration(context, schedule, endpoints,
                                         std::move(declaration));
  if (!normalized)
    return normalized.takeError();
  return MemoryOperationPortRecord(
      std::vector<MemoryTransportEndpointDescriptor>(endpoints.begin(),
                                                     endpoints.end()),
      std::move(*normalized));
}

llvm::Expected<MemoryOperationPortRecord>
MemoryOperationPortRecord::fromCanonical(
    mlir::MLIRContext *context, Schedule schedule,
    llvm::ArrayRef<MemoryTransportEndpointDescriptor> endpoints,
    MemoryOperationPortDeclaration declaration) {
  auto original = detail::encodeMemoryOperationPortDeclaration(declaration);
  if (!original)
    return original.takeError();
  auto normalized =
      create(context, schedule, endpoints, std::move(declaration));
  if (!normalized)
    return normalized.takeError();
  auto canonical = encodeMemoryOperationPortRecord(*normalized);
  if (!canonical)
    return canonical.takeError();
  if (*original != *canonical)
    return invalid("memory operation port record is not canonical");
  return normalized;
}

llvm::Expected<std::vector<MemoryCapabilityMatch>>
MemoryOperationPortRecord::matchingCapabilities(
    const CanonicalActorSchemaProjection &actor,
    const CanonicalService &service,
    const std::optional<CanonicalMemoryAccessView> &access) const {
  if (llvm::Error error = validateCanonicalMemoryActorCorrespondence(
          actor, service, access ? &*access : nullptr))
    return std::move(error);
  auto expectedKind = getMemoryServiceKind(actor.schema);
  if (!expectedKind)
    return expectedKind.takeError();
  if (service.kind() != *expectedKind)
    return invalid("service kind does not match the actor schema");

  std::vector<MemoryCapabilityMatch> matches;
  for (std::uint64_t ordinal = 0;
       ordinal < declaration_.capabilityAlternatives.size(); ++ordinal) {
    const MemoryCapabilityAlternativeRecord &alternative =
        declaration_.capabilityAlternatives[ordinal];
    if (!alternative.actorContractDomain.contains(actor))
      continue;
    const MemoryAccessClass *accessClass = nullptr;
    if (access) {
      if (!alternative.accessDomain)
        continue;
      accessClass = alternative.accessDomain->matchingClass(*access);
      if (!accessClass ||
          !subwordSupportsActual(*accessClass, *access,
                                 alternative.roleToEndpoint, endpoints_))
        continue;
    } else if (alternative.accessDomain) {
      continue;
    }

    const PatternRequirement requirement =
        exactRequirement(actor, access ? &*access : nullptr);
    std::vector<UsePatternKey> patterns;
    for (UsePatternKey pattern : alternative.admissibleUsePatterns)
      if (projectionSatisfies(
              declaration_.operationPatternSemantics[pattern.ordinal()]
                  .transactionProjection,
              requirement))
        patterns.push_back(pattern);
    if (!patterns.empty())
      matches.push_back({ordinal, std::move(patterns)});
  }
  return matches;
}

llvm::Expected<std::optional<std::uint32_t>>
projectMemoryOperationIssueLatency(const MemoryOperationPortRecord &record,
                                   UsePatternKey pattern) {
  if (pattern.ordinal() >= record.resourceContract().usePatternCount())
    return invalid("memory operation issue timing names an absent use pattern");
  return record.resourceContract().usePatternTiming(pattern).commitLatencyCycles;
}

} // namespace fabric
