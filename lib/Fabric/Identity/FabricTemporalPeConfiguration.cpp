#include "Fabric/Identity/FabricTemporalPeConfiguration.h"

#include "Fabric/Identity/FabricRefImport.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace loom::fabric {
namespace {

llvm::Error rejected(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_temporal_pe_configuration_rejected: " +
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

bool rangeIsZero(llvm::ArrayRef<std::uint8_t> bytes, std::uint64_t offset,
                 std::uint64_t width) {
  for (std::uint64_t index = 0; index < width; ++index)
    if (bit(bytes, offset + index))
      return false;
  return true;
}

void writeUnsigned(std::vector<std::uint8_t> &bytes, std::uint64_t offset,
                   std::uint32_t width, std::uint64_t value) {
  for (std::uint32_t index = 0; index < width; ++index)
    setBit(bytes, offset + index, ((value >> index) & 1U) != 0);
}

std::uint64_t readUnsigned(llvm::ArrayRef<std::uint8_t> bytes,
                           std::uint64_t offset, std::uint32_t width) {
  std::uint64_t result = 0;
  for (std::uint32_t index = 0; index < width; ++index)
    result |= static_cast<std::uint64_t>(bit(bytes, offset + index)) << index;
  return result;
}

void writeApInt(std::vector<std::uint8_t> &bytes, std::uint64_t offset,
                const llvm::APInt &value) {
  for (unsigned index = 0; index < value.getBitWidth(); ++index)
    setBit(bytes, offset + index, value[index]);
}

llvm::APInt readApInt(llvm::ArrayRef<std::uint8_t> bytes, std::uint64_t offset,
                      std::uint32_t width) {
  llvm::APInt result(width, 0);
  for (std::uint32_t index = 0; index < width; ++index)
    if (bit(bytes, offset + index))
      result.setBit(index);
  return result;
}

llvm::Error validateCarrier(llvm::ArrayRef<std::uint8_t> bytes,
                            std::uint64_t width) {
  if (bytes.size() != byteCount(width))
    return rejected("direct carrier has the wrong byte count");
  const unsigned used = static_cast<unsigned>(width % 8);
  if (used != 0 && !bytes.empty() &&
      (bytes.back() & static_cast<std::uint8_t>(0xffU << used)) != 0)
    return rejected("direct carrier has nonzero padding bits");
  return llvm::Error::success();
}

llvm::Expected<std::uint64_t>
targetIndex(const FabricTemporalPeSelectorTarget &target,
            std::uint32_t portCount, std::uint32_t fifoCount) {
  if (const auto *port = std::get_if<FabricTemporalPePortTarget>(&target)) {
    if (port->ordinal >= portCount)
      return rejected("selector port target is outside the PE boundary");
    return port->ordinal;
  }
  const auto &fifo = std::get<FabricTemporalPeRegisterFifoTarget>(target);
  if (fifo.ordinal >= fifoCount)
    return rejected("selector register FIFO target is outside the PE");
  return static_cast<std::uint64_t>(portCount) + fifo.ordinal;
}

FabricTemporalPeSelectorTarget decodeTarget(std::uint64_t index,
                                            std::uint32_t portCount) {
  if (index < portCount)
    return FabricTemporalPePortTarget{index};
  return FabricTemporalPeRegisterFifoTarget{index - portCount};
}

template <typename Selection>
llvm::Error encodeSelection(std::vector<std::uint8_t> &bytes,
                            std::uint64_t offset, const Selection &selection,
                            std::uint32_t targetWidth, std::uint32_t portCount,
                            std::uint32_t fifoCount, std::uint32_t tagWidth,
                            bool operand) {
  const auto kind = static_cast<std::uint32_t>(selection.kind);
  if (kind >
      static_cast<std::uint32_t>(FabricTemporalPeSelectorKind::Disconnected))
    return rejected("selector has an unknown variant");
  writeUnsigned(bytes, offset, 2, kind);
  const bool needsTarget =
      selection.kind == FabricTemporalPeSelectorKind::Route ||
      (operand && selection.kind == FabricTemporalPeSelectorKind::Discard);
  if (!needsTarget) {
    if (selection.target || !selection.tag.isZero())
      return rejected("selector carries fields irrelevant to its variant");
    return llvm::Error::success();
  }
  if (!selection.target)
    return rejected("selector omits its required target");
  auto target = targetIndex(*selection.target, portCount, fifoCount);
  if (!target)
    return target.takeError();
  const std::uint64_t targetCount =
      static_cast<std::uint64_t>(portCount) + fifoCount;
  if (*target >= targetCount)
    return rejected("selector target is outside its exact domain");
  if (selection.tag.getBitWidth() != tagWidth)
    return rejected("selector tag has the wrong width");
  writeUnsigned(bytes, offset + 2, targetWidth, *target);
  writeApInt(bytes, offset + 2 + targetWidth, selection.tag);
  return llvm::Error::success();
}

template <typename Selection>
llvm::Expected<Selection>
decodeSelection(llvm::ArrayRef<std::uint8_t> bytes, std::uint64_t offset,
                std::uint32_t targetWidth, std::uint32_t portCount,
                std::uint32_t fifoCount, std::uint32_t tagWidth, bool operand) {
  const std::uint64_t rawKind = readUnsigned(bytes, offset, 2);
  if (rawKind >
      static_cast<std::uint32_t>(FabricTemporalPeSelectorKind::Disconnected))
    return rejected("selector carrier has an unknown variant");
  Selection result;
  result.kind = static_cast<FabricTemporalPeSelectorKind>(rawKind);
  const bool needsTarget =
      result.kind == FabricTemporalPeSelectorKind::Route ||
      (operand && result.kind == FabricTemporalPeSelectorKind::Discard);
  if (!needsTarget) {
    if (!rangeIsZero(bytes, offset + 2, targetWidth + tagWidth))
      return rejected("selector carrier has nonzero irrelevant fields");
    result.target = std::nullopt;
    result.tag = llvm::APInt(tagWidth, 0);
    return result;
  }
  const std::uint64_t target = readUnsigned(bytes, offset + 2, targetWidth);
  const std::uint64_t targetCount =
      static_cast<std::uint64_t>(portCount) + fifoCount;
  if (target >= targetCount)
    return rejected("selector carrier target is outside its exact domain");
  result.target = decodeTarget(target, portCount);
  result.tag = readApInt(bytes, offset + 2 + targetWidth, tagWidth);
  return result;
}

} // namespace

std::uint64_t
FabricTemporalPeConfigurationLayout::rowOffset(std::uint32_t context) const {
  return 1 + static_cast<std::uint64_t>(context) * rowBitCount;
}

std::uint64_t FabricTemporalPeConfigurationLayout::selectedFuOffset(
    std::uint32_t context) const {
  return rowOffset(context) + 1;
}

std::uint64_t FabricTemporalPeConfigurationLayout::operandSelectionOffset(
    std::uint32_t context, std::uint32_t input) const {
  return selectedFuOffset(context) + selectedFuBitCount +
         static_cast<std::uint64_t>(input) * operandSelectionBitCount;
}

std::uint64_t FabricTemporalPeConfigurationLayout::resultSelectionOffset(
    std::uint32_t context, std::uint32_t output) const {
  return operandSelectionOffset(context, maximumFuInputCount) +
         static_cast<std::uint64_t>(output) * resultSelectionBitCount;
}

llvm::Expected<FabricTemporalPeConfigurationSchemaView>
FabricArtifactView::temporalPeConfigurationSchema(
    FabricPeOccurrenceRef occurrence) const {
  if (llvm::Error error = validateFabricRef(*this, occurrence))
    return error;
  if (peSchedule(occurrence) != ::fabric::Schedule::Temporal)
    return rejected("configuration schema requires a Temporal PE");

  FabricTemporalPeConfigurationLayout layout;
  const std::uint64_t contexts = peResidentContextCount(occurrence);
  if (contexts == 0 || contexts > std::numeric_limits<std::uint32_t>::max())
    return rejected("resident context domain is outside u32");
  layout.contextCount = static_cast<std::uint32_t>(contexts);
  const FabricInventoryOwnerRef owner = FabricInventoryOwnerRef::of(occurrence);
  const std::uint64_t fifoCount =
      inventorySize(owner, FabricInventoryKind::RegisterFifo);
  if (fifoCount > std::numeric_limits<std::uint32_t>::max())
    return rejected("register FIFO domain is outside u32");
  layout.registerFifoCount = static_cast<std::uint32_t>(fifoCount);

  const FabricTransportEndpointOwnerRef endpointOwner =
      FabricTransportEndpointOwnerRef::of(occurrence);
  for (FabricOrdinal ordinal = 0;
       ordinal < transportEndpointCount(endpointOwner); ++ordinal) {
    const FabricTransportEndpointRef endpoint{endpointOwner, ordinal};
    const auto direction = transportEndpointDirection(endpoint);
    const auto path = transportEndpointDataPath(endpoint);
    if (!direction || !path || path->kind != ::fabric::DataPathKind::BitsTag ||
        path->tagWidthBits == 0)
      return rejected("Temporal PE endpoint has no tagged data path");
    if (layout.tagWidthBits != 0 && layout.tagWidthBits != path->tagWidthBits)
      return rejected("Temporal PE endpoints disagree on tag width");
    layout.tagWidthBits = path->tagWidthBits;
    if (*direction == FabricPortDirection::Input)
      ++layout.inputPortCount;
    else
      ++layout.outputPortCount;
  }
  if (layout.inputPortCount == 0 || layout.outputPortCount == 0 ||
      layout.tagWidthBits == 0)
    return rejected("Temporal PE boundary shape is empty");

  for (FabricFuOccurrenceRef fu : fuOccurrences()) {
    if (parentPeOf(fu) != occurrence)
      continue;
    const FabricInventoryOwnerRef fuOwner = FabricInventoryOwnerRef::of(fu);
    const std::uint64_t inputs =
        inventorySize(fuOwner, FabricInventoryKind::InputPort);
    const std::uint64_t outputs =
        inventorySize(fuOwner, FabricInventoryKind::OutputPort);
    if (inputs > std::numeric_limits<std::uint32_t>::max() ||
        outputs > std::numeric_limits<std::uint32_t>::max())
      return rejected("Temporal PE FU port domain is outside u32");
    layout.fus.push_back({fu, static_cast<std::uint32_t>(inputs),
                          static_cast<std::uint32_t>(outputs)});
    layout.maximumFuInputCount = std::max(layout.maximumFuInputCount,
                                          static_cast<std::uint32_t>(inputs));
    layout.maximumFuOutputCount = std::max(layout.maximumFuOutputCount,
                                           static_cast<std::uint32_t>(outputs));
  }
  if (layout.fus.empty())
    return rejected("Temporal PE has no FU occurrences");
  layout.selectedFuBitCount = indexWidth(layout.fus.size());
  layout.inputTargetBitCount =
      indexWidth(static_cast<std::uint64_t>(layout.inputPortCount) +
                 layout.registerFifoCount);
  layout.outputTargetBitCount =
      indexWidth(static_cast<std::uint64_t>(layout.outputPortCount) +
                 layout.registerFifoCount);
  layout.operandSelectionBitCount =
      2 + layout.inputTargetBitCount + layout.tagWidthBits;
  layout.resultSelectionBitCount =
      2 + layout.outputTargetBitCount + layout.tagWidthBits;

  const std::uint64_t operandBits =
      static_cast<std::uint64_t>(layout.maximumFuInputCount) *
      layout.operandSelectionBitCount;
  const std::uint64_t resultBits =
      static_cast<std::uint64_t>(layout.maximumFuOutputCount) *
      layout.resultSelectionBitCount;
  if (operandBits > std::numeric_limits<std::uint64_t>::max() - resultBits ||
      1 + layout.selectedFuBitCount >
          std::numeric_limits<std::uint64_t>::max() - operandBits - resultBits)
    return rejected("Temporal PE instruction row is too large");
  layout.rowBitCount = 1 + layout.selectedFuBitCount + operandBits + resultBits;
  if (layout.contextCount >
      (std::numeric_limits<std::uint64_t>::max() - 1) / layout.rowBitCount)
    return rejected("Temporal PE direct carrier is too large");
  layout.carrierBitCount = 1 + layout.contextCount * layout.rowBitCount;

  const FabricSemanticConfigFieldRef field{FabricConfigurationOwnerRef(owner),
                                           0};
  return FabricTemporalPeConfigurationSchemaView(occurrence, field,
                                                 std::move(layout));
}

llvm::Expected<CanonicalSemanticBytes>
FabricTemporalPeConfigurationSchemaView::encode(
    const FabricTemporalPeConfigurationValue &value) const {
  std::vector<std::uint8_t> carrier(byteCount(layout_.carrierBitCount), 0);
  if (std::holds_alternative<FabricTemporalPeDisabled>(value))
    return CanonicalSemanticBytes(std::move(carrier));

  const auto &active = std::get<FabricTemporalPeActive>(value);
  if (active.rows.size() != layout_.contextCount)
    return rejected("Active row count differs from num_instruction");
  if (llvm::none_of(active.rows,
                    [](const auto &row) { return row.has_value(); }))
    return rejected("Active configuration has no active instruction row");
  setBit(carrier, 0, true);

  for (std::uint32_t context = 0; context < layout_.contextCount; ++context) {
    if (!active.rows[context])
      continue;
    const auto &row = *active.rows[context];
    const auto fu = llvm::find_if(layout_.fus, [&](const auto &candidate) {
      return candidate.fu == row.selectedFu;
    });
    if (fu == layout_.fus.end())
      return rejected("instruction row selects a foreign FU");
    const std::uint32_t fuOrdinal =
        static_cast<std::uint32_t>(fu - layout_.fus.begin());
    if (row.operandSelections.size() != fu->inputCount ||
        row.resultSelections.size() != fu->outputCount)
      return rejected("instruction row selector shape differs from its FU");
    setBit(carrier, layout_.rowOffset(context), true);
    writeUnsigned(carrier, layout_.selectedFuOffset(context),
                  layout_.selectedFuBitCount, fuOrdinal);
    for (auto [input, selection] : llvm::enumerate(row.operandSelections))
      if (llvm::Error error = encodeSelection(
              carrier, layout_.operandSelectionOffset(context, input),
              selection, layout_.inputTargetBitCount, layout_.inputPortCount,
              layout_.registerFifoCount, layout_.tagWidthBits, true))
        return std::move(error);
    for (auto [output, selection] : llvm::enumerate(row.resultSelections))
      if (llvm::Error error = encodeSelection(
              carrier, layout_.resultSelectionOffset(context, output),
              selection, layout_.outputTargetBitCount, layout_.outputPortCount,
              layout_.registerFifoCount, layout_.tagWidthBits, false))
        return std::move(error);
  }
  return CanonicalSemanticBytes(std::move(carrier));
}

llvm::Expected<FabricTemporalPeConfigurationValue>
FabricTemporalPeConfigurationSchemaView::decode(
    llvm::ArrayRef<std::uint8_t> bytes) const {
  if (llvm::Error error = validateCarrier(bytes, layout_.carrierBitCount))
    return std::move(error);
  if (!bit(bytes, 0)) {
    if (!rangeIsZero(bytes, 1, layout_.carrierBitCount - 1))
      return rejected("Disabled carrier has nonzero instruction payload");
    return FabricTemporalPeConfigurationValue{FabricTemporalPeDisabled{}};
  }

  FabricTemporalPeActive active;
  active.rows.resize(layout_.contextCount);
  bool hasActiveRow = false;
  for (std::uint32_t context = 0; context < layout_.contextCount; ++context) {
    const std::uint64_t rowOffset = layout_.rowOffset(context);
    if (!bit(bytes, rowOffset)) {
      if (!rangeIsZero(bytes, rowOffset + 1, layout_.rowBitCount - 1))
        return rejected("Unused instruction row has nonzero payload");
      continue;
    }
    hasActiveRow = true;
    const std::uint64_t fuOrdinal = readUnsigned(
        bytes, layout_.selectedFuOffset(context), layout_.selectedFuBitCount);
    if (fuOrdinal >= layout_.fus.size())
      return rejected("instruction row FU index is outside its exact domain");
    const FabricTemporalPeFuShape &fu = layout_.fus[fuOrdinal];
    FabricTemporalPeInstructionEntry row;
    row.selectedFu = fu.fu;
    row.operandSelections.reserve(fu.inputCount);
    row.resultSelections.reserve(fu.outputCount);
    for (std::uint32_t input = 0; input < fu.inputCount; ++input) {
      auto selection = decodeSelection<FabricTemporalPeOperandSelection>(
          bytes, layout_.operandSelectionOffset(context, input),
          layout_.inputTargetBitCount, layout_.inputPortCount,
          layout_.registerFifoCount, layout_.tagWidthBits, true);
      if (!selection)
        return selection.takeError();
      row.operandSelections.push_back(std::move(*selection));
    }
    for (std::uint32_t input = fu.inputCount;
         input < layout_.maximumFuInputCount; ++input)
      if (!rangeIsZero(bytes, layout_.operandSelectionOffset(context, input),
                       layout_.operandSelectionBitCount))
        return rejected("instruction row has nonzero unused input selector");
    for (std::uint32_t output = 0; output < fu.outputCount; ++output) {
      auto selection = decodeSelection<FabricTemporalPeResultSelection>(
          bytes, layout_.resultSelectionOffset(context, output),
          layout_.outputTargetBitCount, layout_.outputPortCount,
          layout_.registerFifoCount, layout_.tagWidthBits, false);
      if (!selection)
        return selection.takeError();
      row.resultSelections.push_back(std::move(*selection));
    }
    for (std::uint32_t output = fu.outputCount;
         output < layout_.maximumFuOutputCount; ++output)
      if (!rangeIsZero(bytes, layout_.resultSelectionOffset(context, output),
                       layout_.resultSelectionBitCount))
        return rejected("instruction row has nonzero unused output selector");
    active.rows[context] = std::move(row);
  }
  if (!hasActiveRow)
    return rejected("Active carrier has no active instruction row");
  auto encoded = encode(FabricTemporalPeConfigurationValue{active});
  if (!encoded)
    return encoded.takeError();
  if (!encoded->bytes().equals(bytes))
    return rejected("Temporal PE carrier does not re-encode canonically");
  return FabricTemporalPeConfigurationValue{std::move(active)};
}

} // namespace loom::fabric
