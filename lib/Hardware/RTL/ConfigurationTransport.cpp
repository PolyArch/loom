#include "Hardware/RTL/ConfigurationTransport.h"

#include "Fabric/Identity/FabricRefBytes.h"
#include "Hardware/RTL/CommonSkeleton.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>

namespace loom::hardware::rtl {
namespace {

using Bytes = std::vector<std::uint8_t>;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "configuration_transport_layout_invalid: " +
                                     message);
}

llvm::Error unsupported(const llvm::Twine &message) {
  return llvm::make_error<FabricStructuralLoweringUnsupportedError>(
      message.str());
}

void appendU64(Bytes &bytes, std::uint64_t value) {
  for (unsigned shift = 64; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> (shift - 8)));
}

void appendFramed(Bytes &bytes, llvm::ArrayRef<std::uint8_t> value) {
  appendU64(bytes, value.size());
  bytes.insert(bytes.end(), value.begin(), value.end());
}

void appendEncoding(Bytes &bytes,
                    const ConfigurationEncodingRelation &relation,
                    const ConfigurationFieldEncoding &field) {
  if (const auto *direct =
          std::get_if<DirectBitsEncoding>(&relation.semanticEncoding)) {
    bytes.push_back(0);
    appendU64(bytes, direct->encodedBitCount);
  } else {
    bytes.push_back(1);
    const auto &codebook =
        std::get<FiniteCodebookEncoding>(relation.semanticEncoding);
    appendU64(bytes, codebook.encodedBitCount);
    appendU64(bytes, codebook.entries.size());
    for (const FiniteCodebookEntry &entry : codebook.entries) {
      appendFramed(bytes, entry.semanticValue);
      appendFramed(bytes, entry.physicalCode);
    }
  }
  appendU64(bytes, field.destinationSlices.size());
  for (const DestinationSlice &slice : field.destinationSlices) {
    appendU64(bytes, slice.sourceBitOffset);
    appendU64(bytes, slice.destinationBitOffset);
    appendU64(bytes, slice.bitCount);
  }
  appendFramed(bytes, relation.inactiveValue);
}

llvm::Expected<Bytes>
definitionKey(const ConfigurationABI &configurationAbi,
              const ProgrammingUnit &unit,
              fabric::SpatialCoreOccurrenceRef spatialCore) {
  std::vector<Bytes> closure;
  closure.reserve(unit.exactFabricResourceClosure.size());
  for (const fabric::FabricPhysicalOccurrenceOwnerRef &owner :
       unit.exactFabricResourceClosure) {
    if (owner.kind() !=
        fabric::FabricPhysicalOccurrenceOwnerKind::SpatialCoreInternal)
      return unsupported(
          "a local programming unit includes a non-SpatialCore owner");
    const auto &internal =
        std::get<fabric::SpatialCoreInternalOccurrenceRef>(owner.payload());
    if (internal.spatialCore != spatialCore)
      return unsupported(
          "a local programming unit spans multiple SpatialCore occurrences");
    closure.push_back(fabric::canonicalFabricBytes(internal.target));
  }
  llvm::sort(closure);

  std::vector<Bytes> fields;
  fields.reserve(unit.fields.size());
  for (const ConfigurationFieldEncoding &field : unit.fields) {
    if (field.slot.kind() !=
        fabric::FabricPhysicalConfigurationSlotKind::SpatialCoreInternalSlot)
      return unsupported(
          "a local programming unit includes a non-SpatialCore field");
    const auto &internal =
        std::get<fabric::SpatialCoreInternalConfigurationSlotRef>(
            field.slot.payload());
    if (internal.spatialCore != spatialCore)
      return unsupported(
          "a local programming unit field names another SpatialCore");
    const ConfigurationEncodingRelation *relation =
        configurationAbi.findEncodingRelation(field);
    if (!relation)
      return invalid("configuration field names an unknown encoding relation");
    Bytes key;
    appendFramed(key, fabric::canonicalFabricBytes(internal.slot));
    appendEncoding(key, *relation, field);
    fields.push_back(std::move(key));
  }
  llvm::sort(fields);

  Bytes result;
  appendU64(result, unit.payloadBitCount);
  appendU64(result, closure.size());
  for (const Bytes &entry : closure)
    appendFramed(result, entry);
  appendU64(result, fields.size());
  for (const Bytes &entry : fields)
    appendFramed(result, entry);
  return result;
}

struct PendingUnit final {
  const ProgrammingUnit *unit = nullptr;
  Bytes definitionKey;
  std::vector<std::uint8_t> inactiveImage;
};

} // namespace

const ConfigurationTransportUnitLayout *
ConfigurationTransportLayout::find(ProgrammingUnitId unitId) const {
  const auto found = llvm::find_if(units, [&](const auto &unit) {
    return unit.programmingUnit.unitId == unitId;
  });
  return found == units.end() ? nullptr : &*found;
}

llvm::Expected<ConfigurationTransportLayout>
derivePortableConfigurationTransportLayout(
    const FinalizedConfigurationABI &configurationAbi,
    fabric::SpatialCoreOccurrenceRef spatialCore) {
  std::vector<PendingUnit> selected;
  for (const ProgrammingUnit &unit :
       configurationAbi.abi().programmingUnits()) {
    const ProgrammingUnitOccurrenceScope scope =
        deriveProgrammingUnitOccurrenceScope(unit);
    if (!llvm::is_contained(scope.spatialCores, spatialCore))
      continue;
    if (scope.spatialCores.size() != 1 ||
        scope.includesDirectSystemResources)
      return unsupported(
          "a programming unit selected by the local transport crosses its "
          "SpatialCore occurrence");

    auto key = definitionKey(configurationAbi.abi(), unit, spatialCore);
    if (!key)
      return key.takeError();
    auto inactive = configurationAbi.abi().encode(unit.id, {});
    if (!inactive)
      return invalid("inactive programming image cannot be encoded: " +
                     llvm::toString(inactive.takeError()));
    selected.push_back(
        PendingUnit{&unit, std::move(*key), std::move(*inactive)});
  }
  llvm::sort(selected, [](const PendingUnit &lhs, const PendingUnit &rhs) {
    return lhs.definitionKey < rhs.definitionKey;
  });
  for (std::size_t index = 1; index < selected.size(); ++index)
    if (selected[index - 1].definitionKey == selected[index].definitionKey)
      return invalid("definition-rebased programming-unit order is ambiguous");

  ConfigurationTransportLayout result;
  result.spatialCore = spatialCore;
  std::uint64_t cursor = 0;
  for (PendingUnit &pending : selected) {
    const ProgrammingUnit &unit = *pending.unit;
    const std::uint64_t payloadBytes = (unit.payloadBitCount + 7) / 8;
    const std::uint64_t payloadWords = (unit.payloadBitCount + 31) / 32;
    if (payloadWords >
        (std::numeric_limits<std::uint64_t>::max() - cursor - 8) / 4)
      return unsupported("configuration transport address span overflows");
    const std::uint64_t commit = cursor + payloadWords * 4;
    const std::uint64_t status = commit + 4;
    const std::uint64_t next = status + 4;
    if (status > std::numeric_limits<std::uint32_t>::max() ||
        next > (std::uint64_t{1} << portableConfigurationAddressWidth))
      return unsupported(
          "configuration transport does not fit the 32-bit address bus");
    result.units.push_back(ConfigurationTransportUnitLayout{
        ProgrammingUnitRef{configurationAbi.reference(), unit.id},
        unit.payloadBitCount, payloadBytes, payloadWords,
        static_cast<std::uint32_t>(cursor), static_cast<std::uint32_t>(commit),
        static_cast<std::uint32_t>(status), std::move(pending.inactiveImage)});
    cursor = next;
  }
  result.byteSpan = cursor;
  return result;
}

} // namespace loom::hardware::rtl
