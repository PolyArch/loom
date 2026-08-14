#include "Fabric/Identity/FabricMemoryConfiguration.h"

#include "Fabric/IR/MemoryActorContractDomain.h"
#include "Fabric/IR/MemoryCapabilityDomains.h"
#include "Fabric/IR/MemoryOperationPort.h"
#include "Fabric/Identity/FabricRefImport.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace loom::fabric {
namespace {

constexpr std::uint32_t roleCount =
    static_cast<std::uint32_t>(
        ::dataflow::semantics::ServiceValueRole::Completion) +
    1;

llvm::Error rejected(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_memory_configuration_rejected: " +
                                     message);
}

std::uint32_t indexWidth(std::uint64_t count) {
  return count <= 1 ? 0 : llvm::Log2_64_Ceil(count);
}

std::uint64_t byteCount(std::uint64_t bits) {
  return bits / 8 + (bits % 8 != 0);
}

bool bit(llvm::ArrayRef<std::uint8_t> bytes, std::uint64_t offset) {
  return ((bytes[static_cast<std::size_t>(offset / 8)] >> (offset % 8)) & 1U) !=
         0;
}

void setBit(std::vector<std::uint8_t> &bytes, std::uint64_t offset,
            bool value) {
  if (!value)
    return;
  bytes[static_cast<std::size_t>(offset / 8)] |=
      static_cast<std::uint8_t>(1U << (offset % 8));
}

void setUnsigned(std::vector<std::uint8_t> &bytes, std::uint64_t offset,
                 std::uint32_t width, std::uint64_t value) {
  for (std::uint32_t selected = 0; selected < width; ++selected)
    setBit(bytes, offset + selected, ((value >> selected) & 1U) != 0);
}

std::uint64_t getUnsigned(llvm::ArrayRef<std::uint8_t> bytes,
                          std::uint64_t offset, std::uint32_t width) {
  std::uint64_t value = 0;
  for (std::uint32_t selected = 0; selected < width; ++selected)
    value |= static_cast<std::uint64_t>(bit(bytes, offset + selected))
             << selected;
  return value;
}

void setApInt(std::vector<std::uint8_t> &bytes, std::uint64_t offset,
              std::uint32_t width, const llvm::APInt &value) {
  for (std::uint32_t selected = 0; selected < width; ++selected)
    setBit(bytes, offset + selected, value[selected]);
}

llvm::APInt getApInt(llvm::ArrayRef<std::uint8_t> bytes, std::uint64_t offset,
                     std::uint32_t width) {
  llvm::APInt value(std::max<std::uint32_t>(1, width), 0);
  for (std::uint32_t selected = 0; selected < width; ++selected)
    value.setBitVal(selected, bit(bytes, offset + selected));
  return value;
}

bool zeroRange(llvm::ArrayRef<std::uint8_t> bytes, std::uint64_t offset,
               std::uint64_t count) {
  for (std::uint64_t selected = 0; selected < count; ++selected)
    if (bit(bytes, offset + selected))
      return false;
  return true;
}

llvm::Expected<std::uint32_t> checkedCount(std::uint64_t value,
                                           llvm::StringRef description) {
  if (value > std::numeric_limits<std::uint32_t>::max())
    return rejected(description + " exceeds the direct-carrier domain");
  return static_cast<std::uint32_t>(value);
}

std::vector<std::uint64_t>
clauseCardinalities(const ::fabric::MemoryActorContractClause &clause) {
  return std::visit(
      [](const auto &selected) -> std::vector<std::uint64_t> {
        using T = std::decay_t<decltype(selected)>;
        if constexpr (std::is_same_v<T,
                                     ::fabric::LoadStorePlainContractClause>) {
          return {selected.volatileValues.size()};
        } else if constexpr (std::is_same_v<
                                 T, ::fabric::LoadStoreAtomicContractClause>) {
          return {selected.orderings.size(), selected.syncScopes.size(),
                  selected.vectorGranularityValues.size(),
                  selected.volatileValues.size()};
        } else if constexpr (std::is_same_v<
                                 T, ::fabric::AtomicRmwContractClause>) {
          return {selected.rmwKinds.size(), selected.orderings.size(),
                  selected.syncScopes.size(),
                  selected.vectorGranularityValues.size(),
                  selected.volatileValues.size()};
        } else if constexpr (std::is_same_v<
                                 T, ::fabric::CompareExchangeContractClause>) {
          return {selected.orderingPairs.size(), selected.syncScopes.size(),
                  selected.vectorGranularityValues.size(),
                  selected.weakValues.size(), selected.volatileValues.size()};
        } else {
          static_assert(std::is_same_v<T, ::fabric::FenceContractClause>);
          return {selected.orderings.size(), selected.syncScopes.size()};
        }
      },
      clause);
}

template <typename T>
std::optional<FabricOrdinal> valueOrdinal(llvm::ArrayRef<T> values,
                                          const T &selected) {
  const auto found = llvm::find(values, selected);
  if (found == values.end())
    return std::nullopt;
  return static_cast<FabricOrdinal>(std::distance(values.begin(), found));
}

std::optional<FabricOrdinal> valueOrdinal(const std::vector<bool> &values,
                                          bool selected) {
  for (FabricOrdinal ordinal = 0; ordinal != values.size(); ++ordinal)
    if (values[ordinal] == selected)
      return ordinal;
  return std::nullopt;
}

template <typename... T>
std::optional<std::vector<FabricOrdinal>> selectedOrdinals(T... values) {
  std::vector<std::optional<FabricOrdinal>> selected{values...};
  if (llvm::any_of(selected,
                   [](const auto &value) { return !value.has_value(); }))
    return std::nullopt;
  std::vector<FabricOrdinal> result;
  result.reserve(selected.size());
  for (const auto &value : selected)
    result.push_back(*value);
  return result;
}

std::optional<std::vector<FabricOrdinal>>
clausePoint(const ::fabric::MemoryActorContractClause &clause,
            const ::dataflow::MemoryContractPayload &payload) {
  return std::visit(
      [&](const auto &projection) -> std::optional<std::vector<FabricOrdinal>> {
        using Projection = std::decay_t<decltype(projection)>;
        if constexpr (std::is_same_v<Projection,
                                     ::dataflow::PlainAccessProjection>) {
          const auto *typed =
              std::get_if<::fabric::LoadStorePlainContractClause>(&clause);
          return typed ? selectedOrdinals(valueOrdinal(typed->volatileValues,
                                                       projection.isVolatile))
                       : std::nullopt;
        } else if constexpr (std::is_same_v<
                                 Projection,
                                 ::dataflow::AtomicAccessProjection>) {
          const auto *typed =
              std::get_if<::fabric::LoadStoreAtomicContractClause>(&clause);
          return typed ? selectedOrdinals(
                             valueOrdinal(llvm::ArrayRef(typed->orderings),
                                          projection.ordering),
                             valueOrdinal(llvm::ArrayRef(typed->syncScopes),
                                          projection.scope),
                             valueOrdinal(
                                 llvm::ArrayRef(typed->vectorGranularityValues),
                                 projection.vectorGranularity),
                             valueOrdinal(typed->volatileValues,
                                          projection.isVolatile))
                       : std::nullopt;
        } else if constexpr (std::is_same_v<Projection,
                                            ::dataflow::AtomicRmwProjection>) {
          const auto *typed =
              std::get_if<::fabric::AtomicRmwContractClause>(&clause);
          return typed ? selectedOrdinals(
                             valueOrdinal(llvm::ArrayRef(typed->rmwKinds),
                                          projection.kind),
                             valueOrdinal(llvm::ArrayRef(typed->orderings),
                                          projection.access.ordering),
                             valueOrdinal(llvm::ArrayRef(typed->syncScopes),
                                          projection.access.scope),
                             valueOrdinal(
                                 llvm::ArrayRef(typed->vectorGranularityValues),
                                 projection.access.vectorGranularity),
                             valueOrdinal(typed->volatileValues,
                                          projection.access.isVolatile))
                       : std::nullopt;
        } else if constexpr (std::is_same_v<
                                 Projection,
                                 ::dataflow::CompareExchangeProjection>) {
          const auto *typed =
              std::get_if<::fabric::CompareExchangeContractClause>(&clause);
          return typed ? selectedOrdinals(
                             valueOrdinal(llvm::ArrayRef(typed->orderingPairs),
                                          ::fabric::CompareExchangeOrderingPair{
                                              projection.successOrdering,
                                              projection.failureOrdering}),
                             valueOrdinal(llvm::ArrayRef(typed->syncScopes),
                                          projection.scope),
                             valueOrdinal(
                                 llvm::ArrayRef(typed->vectorGranularityValues),
                                 projection.vectorGranularity),
                             valueOrdinal(typed->weakValues, projection.weak),
                             valueOrdinal(typed->volatileValues,
                                          projection.isVolatile))
                       : std::nullopt;
        } else {
          static_assert(
              std::is_same_v<Projection, ::dataflow::FenceProjection>);
          const auto *typed =
              std::get_if<::fabric::FenceContractClause>(&clause);
          return typed ? selectedOrdinals(
                             valueOrdinal(llvm::ArrayRef(typed->orderings),
                                          projection.ordering),
                             valueOrdinal(llvm::ArrayRef(typed->syncScopes),
                                          projection.scope))
                       : std::nullopt;
        }
      },
      payload);
}

llvm::Expected<FabricMemoryActorContractSelection>
projectActorContract(const ::fabric::MemoryActorContractDomain &domain,
                     const ::dataflow::CanonicalActorSchemaProjection &actor) {
  const auto *payload =
      std::get_if<::dataflow::MemoryContractPayload>(&actor.payload);
  if (actor.schema != domain.actorSchema() || !payload)
    return rejected("actor does not belong to the selected memory domain");
  std::optional<FabricMemoryActorContractSelection> result;
  for (auto [clauseOrdinal, clause] : llvm::enumerate(domain.clauses())) {
    auto point = clausePoint(clause, *payload);
    if (!point)
      continue;
    if (result)
      return rejected("actor contract has a non-unique domain point");
    result = FabricMemoryActorContractSelection{
        static_cast<FabricOrdinal>(clauseOrdinal), std::move(*point)};
  }
  if (!result)
    return rejected("actor contract is outside the selected memory domain");
  return std::move(*result);
}

llvm::Expected<FabricMemoryAccessSelection>
projectAccess(const ::fabric::ParameterizedMemoryAccessDomain &domain,
              const ::dataflow::semantics::CanonicalMemoryAccessView &access) {
  std::optional<FabricOrdinal> classOrdinal;
  const ::fabric::MemoryAccessClass *selectedClass = nullptr;
  for (auto [ordinal, candidate] : llvm::enumerate(domain.accessClasses())) {
    if (!candidate.contains(access))
      continue;
    if (selectedClass)
      return rejected("memory access has a non-unique capability class");
    classOrdinal = static_cast<FabricOrdinal>(ordinal);
    selectedClass = &candidate;
  }
  if (!selectedClass)
    return rejected("memory access is outside the selected capability");

  ::fabric::InactiveLaneSemantics inactive =
      ::fabric::InactiveLaneSemantics::NotApplicable;
  if (access.maskForm() == ::dataflow::semantics::MemoryMaskForm::Dynamic)
    inactive = access.operation() ==
                       ::dataflow::semantics::MemoryAccessOperation::Store
                   ? ::fabric::InactiveLaneSemantics::Suppress
                   : ::fabric::InactiveLaneSemantics::SuppressAndZeroFill;
  auto maskOrdinal =
      valueOrdinal(selectedClass->maskInactivePairs(),
                   ::fabric::MaskInactivePair{access.maskForm(), inactive});
  const std::optional<std::uint64_t> alignment =
      access.contract().atomic ? access.contract().sourceAlignmentBytes
                               : std::optional<std::uint64_t>(1);
  if (!maskOrdinal || !alignment)
    return rejected("memory access has no exact mask or alignment point");

  FabricOrdinal addressFormat = 0;
  if (access.addressForm() ==
      ::dataflow::semantics::MemoryAddressForm::PointerAddressed) {
    const auto *formats = selectedClass->addressPointerFormats();
    if (!formats || !access.geometry().pointerLayout)
      return rejected("pointer access has no exact address format");
    const ::fabric::PointerFormat expected{
        access.geometry().pointerLayout->addressSpace,
        access.geometry().pointerLayout->representationBits,
        access.geometry().pointerLayout->addressBits,
        access.geometry().pointerLayout->kind};
    auto ordinal = valueOrdinal(formats->formats(), expected);
    if (!ordinal)
      return rejected("pointer access address format is not admitted");
    addressFormat = *ordinal;
  }

  std::optional<FabricOrdinal> dataFormat;
  if (access.geometry().dataPointerLayout) {
    const auto &layout = *access.geometry().dataPointerLayout;
    const ::fabric::PointerFormat expected{layout.addressSpace,
                                           layout.representationBits,
                                           layout.addressBits, layout.kind};
    auto ordinal =
        valueOrdinal(selectedClass->dataPointerFormats().formats(), expected);
    if (!ordinal)
      return rejected("pointer access data format is not admitted");
    dataFormat = *ordinal;
  }

  return FabricMemoryAccessSelection{
      *classOrdinal, access.elementBits(),     access.laneCount(), *maskOrdinal,
      *alignment,    access.addressLaneBits(), addressFormat,      dataFormat};
}

const ::fabric::MemoryRoleEndpointBindingRecord *
findRole(const ::fabric::MemoryCapabilityAlternativeRecord &capability,
         ::dataflow::semantics::ServiceValueRole role) {
  auto found =
      llvm::find_if(capability.roleToEndpoint, [&](const auto &candidate) {
        return candidate.role == role;
      });
  return found == capability.roleToEndpoint.end() ? nullptr : &*found;
}

bool containsRole(llvm::ArrayRef<::dataflow::semantics::ServiceValueRole> roles,
                  ::dataflow::semantics::ServiceValueRole role) {
  return llvm::is_contained(roles, role);
}

bool targetEqual(const ::fabric::MemoryDispatchTarget &lhs,
                 const ::fabric::MemoryDispatchTarget &rhs) {
  return lhs == rhs;
}

bool containsTarget(llvm::ArrayRef<::fabric::MemoryDispatchTarget> targets,
                    const ::fabric::MemoryDispatchTarget &target) {
  return llvm::any_of(targets, [&](const auto &candidate) {
    return targetEqual(candidate, target);
  });
}

std::uint64_t targetCode(const FabricArtifactView &fabric,
                         FabricMemoryOccurrenceRef memory,
                         const ::fabric::MemoryDispatchTarget &target) {
  const bool local = fabric.declaresLocalMemoryService(memory);
  if (std::holds_alternative<::fabric::LocalMemoryDispatchTarget>(target))
    return 0;
  return static_cast<std::uint64_t>(local) +
         std::get<::fabric::ManagerMemoryDispatchTarget>(target)
             .endpointOrdinal;
}

llvm::Expected<::fabric::MemoryDispatchTarget>
decodeTarget(const FabricArtifactView &fabric, FabricMemoryOccurrenceRef memory,
             std::uint64_t code, std::uint32_t managerCount) {
  if (fabric.declaresLocalMemoryService(memory)) {
    if (code == 0)
      return ::fabric::MemoryDispatchTarget(
          std::in_place_type<::fabric::LocalMemoryDispatchTarget>);
    --code;
  }
  if (code >= managerCount)
    return rejected("service-target code is outside its occurrence domain");
  return ::fabric::MemoryDispatchTarget(
      std::in_place_type<::fabric::ManagerMemoryDispatchTarget>,
      ::fabric::ManagerMemoryDispatchTarget{code});
}

std::uint64_t providerMatchBitCount(::fabric::MemoryProviderMatchField field) {
  switch (field) {
  case ::fabric::MemoryProviderMatchField::Range:
    return 128;
  case ::fabric::MemoryProviderMatchField::Prefix:
    return 71;
  case ::fabric::MemoryProviderMatchField::AddressSpace:
    return 32;
  case ::fabric::MemoryProviderMatchField::Context:
    return 64;
  }
  llvm_unreachable("unknown memory provider match field");
}

llvm::Error validateProviderMatch(::fabric::MemoryProviderMatchField field,
                                  const FabricMemoryProviderMatch &match) {
  switch (field) {
  case ::fabric::MemoryProviderMatchField::Range: {
    const auto *range = std::get_if<FabricMemoryRangeMatch>(&match);
    if (!range || range->size == 0 ||
        range->base > std::numeric_limits<std::uint64_t>::max() - range->size)
      return rejected("provider Range match is malformed");
    return llvm::Error::success();
  }
  case ::fabric::MemoryProviderMatchField::Prefix: {
    const auto *prefix = std::get_if<FabricMemoryPrefixMatch>(&match);
    if (!prefix || prefix->prefixLength > 64)
      return rejected("provider Prefix match is malformed");
    const std::uint32_t suffix = 64 - prefix->prefixLength;
    if (suffix == 64 ? prefix->value != 0
                     : suffix != 0 && (prefix->value &
                                       ((std::uint64_t(1) << suffix) - 1)) != 0)
      return rejected("provider Prefix match has noncanonical suffix bits");
    return llvm::Error::success();
  }
  case ::fabric::MemoryProviderMatchField::AddressSpace:
    if (!std::holds_alternative<FabricMemoryAddressSpaceMatch>(match))
      return rejected("provider AddressSpace match has the wrong type");
    return llvm::Error::success();
  case ::fabric::MemoryProviderMatchField::Context:
    if (!std::holds_alternative<FabricMemoryContextMatch>(match))
      return rejected("provider Context match has the wrong type");
    return llvm::Error::success();
  }
  llvm_unreachable("unknown memory provider match field");
}

void encodeProviderMatch(std::vector<std::uint8_t> &bytes, std::uint64_t offset,
                         ::fabric::MemoryProviderMatchField field,
                         const FabricMemoryProviderMatch &match) {
  switch (field) {
  case ::fabric::MemoryProviderMatchField::Range: {
    const auto &range = std::get<FabricMemoryRangeMatch>(match);
    setUnsigned(bytes, offset, 64, range.base);
    setUnsigned(bytes, offset + 64, 64, range.size);
    return;
  }
  case ::fabric::MemoryProviderMatchField::Prefix: {
    const auto &prefix = std::get<FabricMemoryPrefixMatch>(match);
    setUnsigned(bytes, offset, 64, prefix.value);
    setUnsigned(bytes, offset + 64, 7, prefix.prefixLength);
    return;
  }
  case ::fabric::MemoryProviderMatchField::AddressSpace:
    setUnsigned(bytes, offset, 32,
                std::get<FabricMemoryAddressSpaceMatch>(match).addressSpace);
    return;
  case ::fabric::MemoryProviderMatchField::Context:
    setUnsigned(bytes, offset, 64,
                std::get<FabricMemoryContextMatch>(match).context);
    return;
  }
  llvm_unreachable("unknown memory provider match field");
}

FabricMemoryProviderMatch
decodeProviderMatch(llvm::ArrayRef<std::uint8_t> bytes, std::uint64_t offset,
                    ::fabric::MemoryProviderMatchField field) {
  switch (field) {
  case ::fabric::MemoryProviderMatchField::Range:
    return FabricMemoryRangeMatch{getUnsigned(bytes, offset, 64),
                                  getUnsigned(bytes, offset + 64, 64)};
  case ::fabric::MemoryProviderMatchField::Prefix:
    return FabricMemoryPrefixMatch{
        getUnsigned(bytes, offset, 64),
        static_cast<std::uint8_t>(getUnsigned(bytes, offset + 64, 7))};
  case ::fabric::MemoryProviderMatchField::AddressSpace:
    return FabricMemoryAddressSpaceMatch{
        static_cast<std::uint32_t>(getUnsigned(bytes, offset, 32))};
  case ::fabric::MemoryProviderMatchField::Context:
    return FabricMemoryContextMatch{getUnsigned(bytes, offset, 64)};
  }
  llvm_unreachable("unknown memory provider match field");
}

} // namespace

llvm::Expected<FabricMemoryConfigurationSchemaView>
FabricArtifactView::memoryConfigurationSchema(
    FabricMemoryOccurrenceRef memory) const {
  if (llvm::Error error = validateFabricRef(*this, memory))
    return error;
  const auto *connectivity = memoryConnectivity(memory);
  if (!connectivity)
    return rejected("memory occurrence has no connectivity contract");

  FabricMemoryConfigurationLayout layout;
  layout.schedule = memorySchedule(memory);
  layout.roleCount = roleCount;
  auto portCount = checkedCount(memoryOperationPorts(memory).size(),
                                "memory operation-port count");
  if (!portCount)
    return portCount.takeError();
  layout.physicalPortCount = *portCount;
  if (layout.schedule == ::fabric::Schedule::Temporal) {
    auto count = checkedCount(memoryResidentContextCount(memory),
                              "Temporal memory row count");
    if (!count)
      return count.takeError();
    layout.operationRowCount = *count;
  } else if (layout.schedule == ::fabric::Schedule::Spatial) {
    layout.operationRowCount = layout.physicalPortCount;
  } else if (layout.physicalPortCount != 0) {
    return rejected("storage-only memory has operation ports");
  }

  const FabricTransportEndpointOwnerRef transportOwner =
      FabricTransportEndpointOwnerRef::of(memory);
  auto endpointCount = checkedCount(transportEndpointCount(transportOwner),
                                    "memory transport-endpoint count");
  if (!endpointCount)
    return endpointCount.takeError();
  layout.transportEndpointCount = *endpointCount;
  auto connectionCount =
      checkedCount(connectivity->internalConnections().size(),
                   "memory internal-connection count");
  if (!connectionCount)
    return connectionCount.takeError();
  layout.internalConnectionCount = *connectionCount;

  const FabricMemoryEndpointOwnerRef memoryOwner =
      FabricMemoryEndpointOwnerRef::of(memory);
  for (FabricOrdinal ordinal = 0; ordinal < memoryEndpointCount(memoryOwner);
       ++ordinal) {
    const FabricMemoryEndpointRef endpoint{memoryOwner, ordinal};
    if (memoryEndpointRole(endpoint) == FabricMemoryEndpointRole::Manager)
      ++layout.managerEndpointCount;
  }

  std::uint64_t maximumCapabilities = 0;
  std::uint64_t maximumPatterns = 0;
  std::uint64_t maximumClauses = 0;
  std::array<std::uint64_t, 5> maximumClauseValues{};
  std::uint64_t maximumAccessClasses = 0;
  std::uint64_t maximumMaskPairs = 0;
  std::uint64_t maximumPointerFormats = 0;
  for (FabricMemoryOperationPortRef port : memoryOperationPorts(memory)) {
    const auto *record = memoryOperationPort(port);
    if (!record)
      return rejected("memory operation port does not resolve");
    maximumCapabilities = std::max<std::uint64_t>(
        maximumCapabilities, record->capabilityAlternatives().size());
    maximumPatterns = std::max<std::uint64_t>(
        maximumPatterns, record->resourceContract().usePatternCount());
    for (const auto &alternative : record->capabilityAlternatives()) {
      maximumClauses = std::max<std::uint64_t>(
          maximumClauses, alternative.actorContractDomain.clauses().size());
      for (const auto &clause : alternative.actorContractDomain.clauses()) {
        std::vector<std::uint64_t> cardinalities = clauseCardinalities(clause);
        for (auto [ordinal, cardinality] : llvm::enumerate(cardinalities))
          maximumClauseValues[ordinal] =
              std::max(maximumClauseValues[ordinal], cardinality);
      }
      if (!alternative.accessDomain)
        continue;
      maximumAccessClasses = std::max<std::uint64_t>(
          maximumAccessClasses,
          alternative.accessDomain->accessClasses().size());
      for (const auto &access : alternative.accessDomain->accessClasses()) {
        maximumMaskPairs = std::max<std::uint64_t>(
            maximumMaskPairs, access.maskInactivePairs().size());
        if (const auto *formats = access.addressPointerFormats())
          maximumPointerFormats =
              std::max<std::uint64_t>(maximumPointerFormats, formats->size());
        maximumPointerFormats = std::max<std::uint64_t>(
            maximumPointerFormats, access.dataPointerFormats().size());
      }
    }
  }

  layout.physicalPortBitCount = indexWidth(layout.physicalPortCount);
  layout.capabilityBitCount = indexWidth(maximumCapabilities);
  layout.usePatternBitCount = indexWidth(maximumPatterns);
  layout.actorClauseBitCount = indexWidth(maximumClauses);
  for (std::uint64_t cardinality : maximumClauseValues)
    layout.actorValueBitCounts.push_back(indexWidth(cardinality));
  layout.accessClassBitCount = indexWidth(maximumAccessClasses);
  layout.maskPairBitCount = indexWidth(maximumMaskPairs);
  layout.pointerFormatBitCount = indexWidth(maximumPointerFormats);
  layout.transportEndpointBitCount = indexWidth(layout.transportEndpointCount);
  layout.internalConnectionBitCount =
      indexWidth(layout.internalConnectionCount);
  const std::uint64_t targetCount =
      static_cast<std::uint64_t>(declaresLocalMemoryService(memory)) +
      layout.managerEndpointCount;
  if (targetCount == 0 && (layout.operationRowCount != 0 ||
                           !connectivity->subordinateEndpoints().empty()))
    return rejected("memory occurrence has no service-target domain");
  layout.serviceTargetBitCount = indexWidth(targetCount);

  for (FabricOrdinal endpoint = 0; endpoint < layout.transportEndpointCount;
       ++endpoint) {
    const auto path = transportEndpointDataPath(
        FabricTransportEndpointRef{transportOwner, endpoint});
    if (!path)
      return rejected("memory transport endpoint has no data-path type");
    layout.tagWidthBits = std::max(layout.tagWidthBits, path->tagWidthBits);
  }
  if (layout.schedule == ::fabric::Schedule::Spatial &&
      layout.tagWidthBits != 0)
    return rejected("Spatial memory owns tagged transport endpoints");

  layout.roleSourceBitCount = 2 + layout.transportEndpointBitCount +
                              layout.internalConnectionBitCount +
                              layout.tagWidthBits;
  layout.roleDestinationBitCount = 1 + layout.transportEndpointBitCount +
                                   layout.tagWidthBits +
                                   layout.internalConnectionCount;

  std::uint64_t cursor = 1;
  layout.operationRows.reserve(layout.operationRowCount);
  for (std::uint32_t row = 0; row < layout.operationRowCount; ++row) {
    FabricMemoryOperationRowLayout entry;
    entry.bitOffset = cursor++;
    entry.physicalPortOffset = cursor;
    cursor += layout.physicalPortBitCount;
    entry.capabilityOffset = cursor;
    cursor += layout.capabilityBitCount;
    entry.usePatternOffset = cursor;
    cursor += layout.usePatternBitCount;
    entry.actorClauseOffset = cursor;
    cursor += layout.actorClauseBitCount;
    for (std::uint32_t width : layout.actorValueBitCounts) {
      entry.actorValueOffsets.push_back(cursor);
      cursor += width;
    }
    entry.accessPresentOffset = cursor++;
    entry.accessClassOffset = cursor;
    cursor += layout.accessClassBitCount;
    entry.elementWidthOffset = cursor;
    cursor += 64;
    entry.laneCountOffset = cursor;
    cursor += 64;
    entry.maskPairOffset = cursor;
    cursor += layout.maskPairBitCount;
    entry.alignmentOffset = cursor;
    cursor += 64;
    entry.addressLaneWidthOffset = cursor;
    cursor += 32;
    entry.addressPointerFormatOffset = cursor;
    cursor += layout.pointerFormatBitCount;
    entry.dataPointerPresentOffset = cursor++;
    entry.dataPointerFormatOffset = cursor;
    cursor += layout.pointerFormatBitCount;
    entry.baseAddressOffset = cursor;
    cursor += 64;
    for (std::uint32_t role = 0; role < layout.roleCount; ++role) {
      entry.roleSourceOffsets.push_back(cursor);
      cursor += layout.roleSourceBitCount;
    }
    for (std::uint32_t role = 0; role < layout.roleCount; ++role) {
      entry.roleDestinationOffsets.push_back(cursor);
      cursor += layout.roleDestinationBitCount;
    }
    entry.serviceTargetOffset = cursor;
    cursor += layout.serviceTargetBitCount;
    entry.bitCount = cursor - entry.bitOffset;
    layout.operationRows.push_back(std::move(entry));
  }

  layout.providerRows.resize(connectivity->subordinateEndpoints().size());
  for (auto [endpoint, subordinate] :
       llvm::enumerate(connectivity->subordinateEndpoints())) {
    auto rowCount = checkedCount(subordinate.maxExposedBindings,
                                 "subordinate provider row count");
    if (!rowCount)
      return rowCount.takeError();
    auto &rows = layout.providerRows[endpoint];
    rows.reserve(*rowCount);
    for (std::uint32_t row = 0; row < *rowCount; ++row) {
      FabricMemoryProviderRowLayout entry;
      entry.bitOffset = cursor++;
      for (::fabric::MemoryProviderMatchField field : subordinate.matchFields) {
        entry.matchOffsets.push_back(cursor);
        cursor += providerMatchBitCount(field);
      }
      entry.serviceTargetOffset = cursor;
      cursor += layout.serviceTargetBitCount;
      entry.baseOffsetOffset = cursor;
      if (subordinate.addressTransform ==
          ::fabric::MemoryProviderAddressTransform::ConstantBaseOffset)
        cursor += 64;
      entry.bitCount = cursor - entry.bitOffset;
      rows.push_back(std::move(entry));
    }
  }
  layout.carrierBitCount = cursor;
  if (layout.carrierBitCount == 0 ||
      byteCount(layout.carrierBitCount) >
          std::numeric_limits<std::size_t>::max())
    return rejected("memory direct carrier is too large for this host");

  const FabricConfigurationOwnerRef owner(FabricInventoryOwnerRef::of(memory));
  return FabricMemoryConfigurationSchemaView(
      this, memory, FabricSemanticConfigFieldRef{owner, 0}, std::move(layout));
}

llvm::Expected<FabricMemoryOperationRow>
FabricMemoryConfigurationSchemaView::projectOperationRow(
    FabricOrdinal physicalPort, FabricOrdinal capabilityAlternative,
    FabricOrdinal usePattern,
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    const std::optional<::dataflow::semantics::CanonicalMemoryAccessView>
        &access,
    std::uint64_t baseAddressBytes,
    std::vector<std::optional<FabricMemoryRoleSource>> roleSources,
    std::vector<std::optional<FabricMemoryRoleDestination>> roleDestinations,
    ::fabric::MemoryDispatchTarget serviceTarget) const {
  if (physicalPort >= layout_.physicalPortCount)
    return rejected("operation projection selects an invalid physical port");
  const FabricMemoryOperationPortRef port{memory_, physicalPort};
  const auto *portRecord = fabric_->memoryOperationPort(port);
  if (!portRecord ||
      capabilityAlternative >= portRecord->capabilityAlternatives().size())
    return rejected("operation projection selects an invalid capability");
  const auto &capability =
      portRecord->capabilityAlternatives()[capabilityAlternative];
  if (usePattern >= portRecord->resourceContract().usePatternCount() ||
      !llvm::is_contained(capability.admissibleUsePatterns,
                          ::fabric::UsePatternKey(usePattern)))
    return rejected("operation projection selects an invalid use pattern");

  auto actorContract =
      projectActorContract(capability.actorContractDomain, actor);
  if (!actorContract)
    return actorContract.takeError();
  std::optional<FabricMemoryAccessSelection> accessSelection;
  if (capability.accessDomain.has_value() != access.has_value())
    return rejected("operation projection has inconsistent access presence");
  if (access) {
    auto selected = projectAccess(*capability.accessDomain, *access);
    if (!selected)
      return selected.takeError();
    accessSelection = std::move(*selected);
  }

  FabricMemoryOperationRow result{physicalPort,
                                  capabilityAlternative,
                                  usePattern,
                                  std::move(*actorContract),
                                  std::move(accessSelection),
                                  baseAddressBytes,
                                  std::move(roleSources),
                                  std::move(roleDestinations),
                                  std::move(serviceTarget)};

  FabricMemoryActive active;
  active.operationRows.resize(layout_.operationRows.size());
  active.providerDecodeRows.resize(layout_.providerRows.size());
  for (auto [ordinal, rows] : llvm::enumerate(layout_.providerRows))
    active.providerDecodeRows[ordinal].resize(rows.size());
  const FabricOrdinal rowOrdinal =
      layout_.schedule == ::fabric::Schedule::Spatial
          ? physicalPort
          : std::numeric_limits<FabricOrdinal>::max();
  if (rowOrdinal != std::numeric_limits<FabricOrdinal>::max()) {
    active.operationRows[rowOrdinal] = result;
    auto encoded = encode(FabricMemoryConfigurationValue{active});
    if (!encoded)
      return encoded.takeError();
  }
  return result;
}

llvm::Expected<CanonicalSemanticBytes>
FabricMemoryConfigurationSchemaView::encode(
    const FabricMemoryConfigurationValue &value) const {
  std::vector<std::uint8_t> bytes(
      static_cast<std::size_t>(byteCount(layout_.carrierBitCount)), 0);
  if (std::holds_alternative<FabricMemoryDisabled>(value))
    return CanonicalSemanticBytes(std::move(bytes));
  const auto &active = std::get<FabricMemoryActive>(value);
  const auto *connectivity = fabric_->memoryConnectivity(memory_);
  if (!connectivity ||
      active.operationRows.size() != layout_.operationRows.size() ||
      active.providerDecodeRows.size() != layout_.providerRows.size())
    return rejected("active memory configuration has the wrong row shape");

  bool anyActive = false;
  std::set<std::pair<FabricOrdinal, std::vector<std::uint64_t>>>
      temporalInputMatches;
  for (auto [rowOrdinal, selected] : llvm::enumerate(active.operationRows)) {
    const auto &rowLayout = layout_.operationRows[rowOrdinal];
    if (!selected)
      continue;
    anyActive = true;
    const FabricMemoryOperationRow &row = *selected;
    const FabricOrdinal portOrdinal =
        layout_.schedule == ::fabric::Schedule::Spatial
            ? static_cast<FabricOrdinal>(rowOrdinal)
            : row.physicalPort;
    if (row.physicalPort != portOrdinal ||
        portOrdinal >= layout_.physicalPortCount)
      return rejected("operation row selects an invalid physical port");
    const FabricMemoryOperationPortRef port{memory_, portOrdinal};
    const auto *portRecord = fabric_->memoryOperationPort(port);
    if (!portRecord || row.capabilityAlternative >=
                           portRecord->capabilityAlternatives().size())
      return rejected("operation row selects an invalid capability");
    const auto &capability =
        portRecord->capabilityAlternatives()[row.capabilityAlternative];
    if (row.usePattern >= portRecord->resourceContract().usePatternCount() ||
        !llvm::is_contained(capability.admissibleUsePatterns,
                            ::fabric::UsePatternKey(row.usePattern)))
      return rejected("operation row selects an inadmissible use pattern");

    const auto clauses = capability.actorContractDomain.clauses();
    if (row.actorContract.clause >= clauses.size())
      return rejected("operation row selects an invalid actor clause");
    const auto cardinalities =
        clauseCardinalities(clauses[row.actorContract.clause]);
    if (row.actorContract.values.size() != cardinalities.size())
      return rejected("operation row actor-clause point has the wrong shape");
    for (auto [ordinal, selectedValue] :
         llvm::enumerate(row.actorContract.values))
      if (selectedValue >= cardinalities[ordinal])
        return rejected("operation row actor-clause value is out of range");

    if (capability.accessDomain.has_value() != row.access.has_value())
      return rejected(
          "operation row access presence disagrees with capability");
    const ::fabric::MemoryAccessClass *accessClass = nullptr;
    ::dataflow::semantics::MemoryMaskForm maskForm =
        ::dataflow::semantics::MemoryMaskForm::Absent;
    if (row.access) {
      const auto classes = capability.accessDomain->accessClasses();
      if (row.access->accessClass >= classes.size())
        return rejected("operation row selects an invalid access class");
      accessClass = &classes[row.access->accessClass];
      if (!accessClass->elementWidths().contains(
              row.access->elementWidthBits) ||
          !accessClass->flattenedLaneCounts().contains(
              row.access->flattenedLaneCount) ||
          row.access->maskInactivePair >=
              accessClass->maskInactivePairs().size() ||
          !accessClass->sourceAlignments().containsBytes(
              row.access->sourceAlignmentBytes))
        return rejected("operation row access point is outside its domains");
      maskForm =
          accessClass->maskInactivePairs()[row.access->maskInactivePair].mask;
      if (const auto *widths = accessClass->rootRelativeIndexWidths()) {
        if (!widths->contains(row.access->addressLaneWidthBits) ||
            row.access->addressPointerFormat != 0)
          return rejected("root-relative access has an invalid address point");
      } else {
        const auto *formats = accessClass->addressPointerFormats();
        if (!formats ||
            row.access->addressPointerFormat >= formats->formats().size() ||
            row.access->addressLaneWidthBits !=
                formats->formats()[row.access->addressPointerFormat]
                    .representationBits)
          return rejected("pointer access has an invalid address format");
      }
      if (row.access->dataPointerFormat &&
          *row.access->dataPointerFormat >=
              accessClass->dataPointerFormats().formats().size())
        return rejected("access selects an invalid data pointer format");
    }

    auto kind = ::dataflow::semantics::getMemoryServiceKind(
        capability.actorContractDomain.actorSchema());
    if (!kind)
      return kind.takeError();
    const auto &roleSchema = ::dataflow::semantics::getServiceRoleSchema(*kind);
    if (row.roleSources.size() != layout_.roleCount ||
        row.roleDestinations.size() != layout_.roleCount)
      return rejected("operation row role vectors have the wrong shape");

    for (std::uint32_t roleOrdinal = 0; roleOrdinal < layout_.roleCount;
         ++roleOrdinal) {
      const auto role =
          static_cast<::dataflow::semantics::ServiceValueRole>(roleOrdinal);
      bool inputActive = containsRole(roleSchema.arguments, role);
      if (role == ::dataflow::semantics::ServiceValueRole::Mask &&
          maskForm == ::dataflow::semantics::MemoryMaskForm::Absent)
        inputActive = false;
      const bool outputActive = containsRole(roleSchema.results, role);
      const auto *binding = findRole(capability, role);
      if ((inputActive || outputActive) && !binding)
        return rejected("active memory role has no capability endpoint");

      const auto &source = row.roleSources[roleOrdinal];
      if (source.has_value() != inputActive)
        return rejected("operation row input-role source is inconsistent");
      if (source) {
        if (const auto *external =
                std::get_if<FabricMemoryExternalRoleSource>(&*source)) {
          if (external->endpoint != binding->endpointOrdinal ||
              external->endpoint >= layout_.transportEndpointCount)
            return rejected("external input role selects the wrong endpoint");
          if (layout_.tagWidthBits == 0) {
            if (!external->tag.isZero())
              return rejected("Spatial memory input carries a tag");
          } else if (external->tag.getBitWidth() != layout_.tagWidthBits) {
            return rejected("Temporal memory input tag has the wrong width");
          }
          if (layout_.schedule == ::fabric::Schedule::Temporal) {
            std::vector<std::uint64_t> tagWords;
            const unsigned wordCount = external->tag.getNumWords();
            tagWords.assign(external->tag.getRawData(),
                            external->tag.getRawData() + wordCount);
            if (!temporalInputMatches
                     .insert({external->endpoint, std::move(tagWords)})
                     .second) {
              std::string tagText;
              llvm::raw_string_ostream tagStream(tagText);
              external->tag.print(tagStream, /*isSigned=*/false);
              return rejected(
                  "Temporal memory repeats an ingress tag match: endpoint " +
                  llvm::Twine(external->endpoint) + " role " +
                  llvm::Twine(roleOrdinal) + " tag " + tagText +
                  " is already matched by another configured operation row");
            }
          }
        } else {
          const FabricOrdinal connection =
              std::get<FabricMemoryInternalRoleSource>(*source).connection;
          if (connection >= connectivity->internalConnections().size() ||
              connectivity->internalConnections()[connection]
                      .sinkEndpointOrdinal != binding->endpointOrdinal)
            return rejected("internal input role selects an ineligible edge");
        }
      }

      const auto &destination = row.roleDestinations[roleOrdinal];
      if (destination.has_value() != outputActive)
        return rejected(
            "operation row output-role destination is inconsistent");
      if (!destination)
        continue;
      if (!destination->external && destination->internalConnections.empty())
        return rejected("active output role has no destination");
      if (destination->external) {
        if (destination->external->endpoint != binding->endpointOrdinal ||
            destination->external->endpoint >= layout_.transportEndpointCount)
          return rejected("external output role selects the wrong endpoint");
        if (layout_.tagWidthBits == 0) {
          if (!destination->external->tag.isZero())
            return rejected("Spatial memory output carries a tag");
        } else if (destination->external->tag.getBitWidth() !=
                   layout_.tagWidthBits) {
          return rejected("Temporal memory output tag has the wrong width");
        }
      }
      FabricOrdinal previous = 0;
      bool hasPrevious = false;
      for (FabricOrdinal connection : destination->internalConnections) {
        if (connection >= connectivity->internalConnections().size() ||
            connectivity->internalConnections()[connection]
                    .sourceEndpointOrdinal != binding->endpointOrdinal ||
            (hasPrevious && connection <= previous))
          return rejected("internal output destination is noncanonical");
        previous = connection;
        hasPrevious = true;
      }
    }

    const auto &targets =
        connectivity->operationPorts()[portOrdinal]
            .capabilityTargetDomains[row.capabilityAlternative];
    if (!containsTarget(targets, row.serviceTarget))
      return rejected("operation row service target is outside H_dispatch");

    setBit(bytes, rowLayout.bitOffset, true);
    setUnsigned(bytes, rowLayout.physicalPortOffset,
                layout_.physicalPortBitCount, row.physicalPort);
    setUnsigned(bytes, rowLayout.capabilityOffset, layout_.capabilityBitCount,
                row.capabilityAlternative);
    setUnsigned(bytes, rowLayout.usePatternOffset, layout_.usePatternBitCount,
                row.usePattern);
    setUnsigned(bytes, rowLayout.actorClauseOffset, layout_.actorClauseBitCount,
                row.actorContract.clause);
    for (auto [ordinal, selectedValue] :
         llvm::enumerate(row.actorContract.values))
      setUnsigned(bytes, rowLayout.actorValueOffsets[ordinal],
                  layout_.actorValueBitCounts[ordinal], selectedValue);
    if (row.access) {
      setBit(bytes, rowLayout.accessPresentOffset, true);
      setUnsigned(bytes, rowLayout.accessClassOffset,
                  layout_.accessClassBitCount, row.access->accessClass);
      setUnsigned(bytes, rowLayout.elementWidthOffset, 64,
                  row.access->elementWidthBits);
      setUnsigned(bytes, rowLayout.laneCountOffset, 64,
                  row.access->flattenedLaneCount);
      setUnsigned(bytes, rowLayout.maskPairOffset, layout_.maskPairBitCount,
                  row.access->maskInactivePair);
      setUnsigned(bytes, rowLayout.alignmentOffset, 64,
                  row.access->sourceAlignmentBytes);
      setUnsigned(bytes, rowLayout.addressLaneWidthOffset, 32,
                  row.access->addressLaneWidthBits);
      setUnsigned(bytes, rowLayout.addressPointerFormatOffset,
                  layout_.pointerFormatBitCount,
                  row.access->addressPointerFormat);
      if (row.access->dataPointerFormat) {
        setBit(bytes, rowLayout.dataPointerPresentOffset, true);
        setUnsigned(bytes, rowLayout.dataPointerFormatOffset,
                    layout_.pointerFormatBitCount,
                    *row.access->dataPointerFormat);
      }
    }
    setUnsigned(bytes, rowLayout.baseAddressOffset, 64, row.baseAddressBytes);
    for (std::uint32_t role = 0; role < layout_.roleCount; ++role) {
      if (row.roleSources[role]) {
        const std::uint64_t offset = rowLayout.roleSourceOffsets[role];
        setBit(bytes, offset, true);
        if (const auto *external = std::get_if<FabricMemoryExternalRoleSource>(
                &*row.roleSources[role])) {
          setUnsigned(bytes, offset + 2, layout_.transportEndpointBitCount,
                      external->endpoint);
          setApInt(bytes,
                   offset + 2 + layout_.transportEndpointBitCount +
                       layout_.internalConnectionBitCount,
                   layout_.tagWidthBits, external->tag);
        } else {
          setBit(bytes, offset + 1, true);
          setUnsigned(
              bytes, offset + 2 + layout_.transportEndpointBitCount,
              layout_.internalConnectionBitCount,
              std::get<FabricMemoryInternalRoleSource>(*row.roleSources[role])
                  .connection);
        }
      }
      if (row.roleDestinations[role]) {
        const std::uint64_t offset = rowLayout.roleDestinationOffsets[role];
        if (row.roleDestinations[role]->external) {
          setBit(bytes, offset, true);
          setUnsigned(bytes, offset + 1, layout_.transportEndpointBitCount,
                      row.roleDestinations[role]->external->endpoint);
          setApInt(bytes, offset + 1 + layout_.transportEndpointBitCount,
                   layout_.tagWidthBits,
                   row.roleDestinations[role]->external->tag);
        }
        const std::uint64_t connectionBase = offset + 1 +
                                             layout_.transportEndpointBitCount +
                                             layout_.tagWidthBits;
        for (FabricOrdinal connection :
             row.roleDestinations[role]->internalConnections)
          setBit(bytes, connectionBase + connection, true);
      }
    }
    setUnsigned(bytes, rowLayout.serviceTargetOffset,
                layout_.serviceTargetBitCount,
                targetCode(*fabric_, memory_, row.serviceTarget));
  }

  for (auto [endpointOrdinal, selectedRows] :
       llvm::enumerate(active.providerDecodeRows)) {
    const auto &subordinate =
        connectivity->subordinateEndpoints()[endpointOrdinal];
    const auto &rowLayouts = layout_.providerRows[endpointOrdinal];
    if (selectedRows.size() != rowLayouts.size())
      return rejected("provider decode table has the wrong row count");
    for (auto [rowOrdinal, selected] : llvm::enumerate(selectedRows)) {
      if (!selected)
        continue;
      anyActive = true;
      const auto &row = *selected;
      const auto &rowLayout = rowLayouts[rowOrdinal];
      if (row.matches.size() != subordinate.matchFields.size())
        return rejected("provider decode row has the wrong match shape");
      for (auto [matchOrdinal, match] : llvm::enumerate(row.matches)) {
        const auto field = subordinate.matchFields[matchOrdinal];
        if (llvm::Error error = validateProviderMatch(field, match))
          return error;
        encodeProviderMatch(bytes, rowLayout.matchOffsets[matchOrdinal], field,
                            match);
      }
      if (!containsTarget(subordinate.targetDomain, row.serviceTarget))
        return rejected("provider decode target is outside H_dispatch");
      if (subordinate.addressTransform ==
              ::fabric::MemoryProviderAddressTransform::None &&
          row.baseOffsetBytes != 0)
        return rejected("provider decode has an undeclared base transform");
      setBit(bytes, rowLayout.bitOffset, true);
      setUnsigned(bytes, rowLayout.serviceTargetOffset,
                  layout_.serviceTargetBitCount,
                  targetCode(*fabric_, memory_, row.serviceTarget));
      if (subordinate.addressTransform ==
          ::fabric::MemoryProviderAddressTransform::ConstantBaseOffset)
        setUnsigned(bytes, rowLayout.baseOffsetOffset, 64, row.baseOffsetBytes);
    }
  }
  if (!anyActive)
    return rejected("empty Active memory configuration is not canonical");
  setBit(bytes, 0, true);
  return CanonicalSemanticBytes(std::move(bytes));
}

llvm::Expected<FabricMemoryConfigurationValue>
FabricMemoryConfigurationSchemaView::decode(
    llvm::ArrayRef<std::uint8_t> bytes) const {
  if (bytes.size() != byteCount(layout_.carrierBitCount))
    return rejected("direct carrier has the wrong byte count");
  const unsigned usedBits = layout_.carrierBitCount % 8;
  if (usedBits != 0 &&
      (bytes.back() & static_cast<std::uint8_t>(0xffU << usedBits)) != 0)
    return rejected("direct carrier has nonzero padding bits");
  if (!bit(bytes, 0)) {
    if (!zeroRange(bytes, 1, layout_.carrierBitCount - 1))
      return rejected("Disabled memory carrier has nonzero payload");
    return FabricMemoryConfigurationValue{FabricMemoryDisabled{}};
  }

  const auto *connectivity = fabric_->memoryConnectivity(memory_);
  if (!connectivity)
    return rejected("memory occurrence lost its connectivity contract");
  FabricMemoryActive active;
  active.operationRows.resize(layout_.operationRows.size());
  for (auto [rowOrdinal, rowLayout] : llvm::enumerate(layout_.operationRows)) {
    if (!bit(bytes, rowLayout.bitOffset)) {
      if (!zeroRange(bytes, rowLayout.bitOffset + 1, rowLayout.bitCount - 1))
        return rejected("unused operation row has nonzero payload");
      continue;
    }
    FabricMemoryOperationRow row;
    row.physicalPort = layout_.schedule == ::fabric::Schedule::Spatial
                           ? static_cast<FabricOrdinal>(rowOrdinal)
                           : getUnsigned(bytes, rowLayout.physicalPortOffset,
                                         layout_.physicalPortBitCount);
    row.capabilityAlternative = getUnsigned(bytes, rowLayout.capabilityOffset,
                                            layout_.capabilityBitCount);
    row.usePattern = getUnsigned(bytes, rowLayout.usePatternOffset,
                                 layout_.usePatternBitCount);
    row.actorContract.clause = getUnsigned(bytes, rowLayout.actorClauseOffset,
                                           layout_.actorClauseBitCount);
    if (row.physicalPort >= layout_.physicalPortCount)
      return rejected("decoded operation row selects an invalid port");
    const auto *port = fabric_->memoryOperationPort(
        FabricMemoryOperationPortRef{memory_, row.physicalPort});
    if (!port ||
        row.capabilityAlternative >= port->capabilityAlternatives().size())
      return rejected("decoded operation row selects an invalid capability");
    const auto &capability =
        port->capabilityAlternatives()[row.capabilityAlternative];
    if (row.actorContract.clause >=
        capability.actorContractDomain.clauses().size())
      return rejected("decoded operation row selects an invalid clause");
    const auto cardinalities = clauseCardinalities(
        capability.actorContractDomain.clauses()[row.actorContract.clause]);
    for (auto [ordinal, cardinality] : llvm::enumerate(cardinalities)) {
      const FabricOrdinal selected =
          getUnsigned(bytes, rowLayout.actorValueOffsets[ordinal],
                      layout_.actorValueBitCounts[ordinal]);
      if (selected >= cardinality)
        return rejected("decoded actor-clause value is out of range");
      row.actorContract.values.push_back(selected);
    }
    for (std::size_t ordinal = cardinalities.size();
         ordinal < rowLayout.actorValueOffsets.size(); ++ordinal)
      if (getUnsigned(bytes, rowLayout.actorValueOffsets[ordinal],
                      layout_.actorValueBitCounts[ordinal]) != 0)
        return rejected("unused actor-clause slot is nonzero");

    if (bit(bytes, rowLayout.accessPresentOffset)) {
      row.access = FabricMemoryAccessSelection{
          getUnsigned(bytes, rowLayout.accessClassOffset,
                      layout_.accessClassBitCount),
          getUnsigned(bytes, rowLayout.elementWidthOffset, 64),
          getUnsigned(bytes, rowLayout.laneCountOffset, 64),
          getUnsigned(bytes, rowLayout.maskPairOffset,
                      layout_.maskPairBitCount),
          getUnsigned(bytes, rowLayout.alignmentOffset, 64),
          static_cast<std::uint32_t>(
              getUnsigned(bytes, rowLayout.addressLaneWidthOffset, 32)),
          getUnsigned(bytes, rowLayout.addressPointerFormatOffset,
                      layout_.pointerFormatBitCount),
          std::nullopt};
      if (bit(bytes, rowLayout.dataPointerPresentOffset))
        row.access->dataPointerFormat =
            getUnsigned(bytes, rowLayout.dataPointerFormatOffset,
                        layout_.pointerFormatBitCount);
    } else {
      const std::uint64_t accessBegin = rowLayout.accessClassOffset;
      const std::uint64_t accessEnd = rowLayout.baseAddressOffset;
      if (!zeroRange(bytes, accessBegin, accessEnd - accessBegin))
        return rejected("absent access projection has nonzero payload");
    }
    row.baseAddressBytes = getUnsigned(bytes, rowLayout.baseAddressOffset, 64);
    row.roleSources.resize(layout_.roleCount);
    row.roleDestinations.resize(layout_.roleCount);
    for (std::uint32_t role = 0; role < layout_.roleCount; ++role) {
      const std::uint64_t sourceOffset = rowLayout.roleSourceOffsets[role];
      if (bit(bytes, sourceOffset)) {
        if (bit(bytes, sourceOffset + 1)) {
          const FabricOrdinal connection = getUnsigned(
              bytes, sourceOffset + 2 + layout_.transportEndpointBitCount,
              layout_.internalConnectionBitCount);
          if (getUnsigned(bytes, sourceOffset + 2,
                          layout_.transportEndpointBitCount) != 0 ||
              !getApInt(bytes,
                        sourceOffset + 2 + layout_.transportEndpointBitCount +
                            layout_.internalConnectionBitCount,
                        layout_.tagWidthBits)
                   .isZero())
            return rejected("internal source has nonzero external payload");
          row.roleSources[role] = FabricMemoryInternalRoleSource{connection};
        } else {
          if (getUnsigned(bytes,
                          sourceOffset + 2 + layout_.transportEndpointBitCount,
                          layout_.internalConnectionBitCount) != 0)
            return rejected("external source has nonzero internal selector");
          row.roleSources[role] = FabricMemoryExternalRoleSource{
              getUnsigned(bytes, sourceOffset + 2,
                          layout_.transportEndpointBitCount),
              getApInt(bytes,
                       sourceOffset + 2 + layout_.transportEndpointBitCount +
                           layout_.internalConnectionBitCount,
                       layout_.tagWidthBits)};
        }
      } else if (!zeroRange(bytes, sourceOffset + 1,
                            layout_.roleSourceBitCount - 1)) {
        return rejected("absent role source has nonzero payload");
      }

      const std::uint64_t destinationOffset =
          rowLayout.roleDestinationOffsets[role];
      FabricMemoryRoleDestination destination;
      if (bit(bytes, destinationOffset))
        destination.external = FabricMemoryExternalRoleSource{
            getUnsigned(bytes, destinationOffset + 1,
                        layout_.transportEndpointBitCount),
            getApInt(bytes,
                     destinationOffset + 1 + layout_.transportEndpointBitCount,
                     layout_.tagWidthBits)};
      else if (getUnsigned(bytes, destinationOffset + 1,
                           layout_.transportEndpointBitCount) != 0 ||
               !getApInt(bytes,
                         destinationOffset + 1 +
                             layout_.transportEndpointBitCount,
                         layout_.tagWidthBits)
                    .isZero())
        return rejected("absent external destination has nonzero payload");
      const std::uint64_t connectionBase = destinationOffset + 1 +
                                           layout_.transportEndpointBitCount +
                                           layout_.tagWidthBits;
      for (FabricOrdinal connection = 0;
           connection < layout_.internalConnectionCount; ++connection)
        if (bit(bytes, connectionBase + connection))
          destination.internalConnections.push_back(connection);
      if (destination.external || !destination.internalConnections.empty())
        row.roleDestinations[role] = std::move(destination);
    }
    auto target = decodeTarget(*fabric_, memory_,
                               getUnsigned(bytes, rowLayout.serviceTargetOffset,
                                           layout_.serviceTargetBitCount),
                               layout_.managerEndpointCount);
    if (!target)
      return target.takeError();
    row.serviceTarget = std::move(*target);
    active.operationRows[rowOrdinal] = std::move(row);
  }

  active.providerDecodeRows.resize(layout_.providerRows.size());
  for (auto [endpointOrdinal, rows] : llvm::enumerate(layout_.providerRows)) {
    const auto &subordinate =
        connectivity->subordinateEndpoints()[endpointOrdinal];
    auto &decodedRows = active.providerDecodeRows[endpointOrdinal];
    decodedRows.resize(rows.size());
    for (auto [rowOrdinal, rowLayout] : llvm::enumerate(rows)) {
      if (!bit(bytes, rowLayout.bitOffset)) {
        if (!zeroRange(bytes, rowLayout.bitOffset + 1, rowLayout.bitCount - 1))
          return rejected("unused provider row has nonzero payload");
        continue;
      }
      FabricMemoryProviderDecodeRow row;
      for (auto [matchOrdinal, field] :
           llvm::enumerate(subordinate.matchFields))
        row.matches.push_back(decodeProviderMatch(
            bytes, rowLayout.matchOffsets[matchOrdinal], field));
      auto target =
          decodeTarget(*fabric_, memory_,
                       getUnsigned(bytes, rowLayout.serviceTargetOffset,
                                   layout_.serviceTargetBitCount),
                       layout_.managerEndpointCount);
      if (!target)
        return target.takeError();
      row.serviceTarget = std::move(*target);
      if (subordinate.addressTransform ==
          ::fabric::MemoryProviderAddressTransform::ConstantBaseOffset)
        row.baseOffsetBytes =
            getUnsigned(bytes, rowLayout.baseOffsetOffset, 64);
      decodedRows[rowOrdinal] = std::move(row);
    }
  }

  auto canonical = encode(FabricMemoryConfigurationValue{active});
  if (!canonical)
    return canonical.takeError();
  if (!canonical->bytes().equals(bytes))
    return rejected("memory direct carrier is not canonical");
  return FabricMemoryConfigurationValue{std::move(active)};
}

} // namespace loom::fabric
