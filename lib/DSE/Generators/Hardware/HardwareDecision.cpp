#include "DSE/HardwareDecision.h"

#include "Common/ArtifactLocalReference.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace loom::dse {

llvm::StringRef
hardwareMappingImpactKindSpelling(HardwareMappingImpactKind kind) {
  switch (kind) {
  case HardwareMappingImpactKind::Unchanged:
    return "unchanged";
  case HardwareMappingImpactKind::Rebase:
    return "rebase";
  case HardwareMappingImpactKind::Reopen:
    return "reopen";
  }
  llvm_unreachable("unknown hardware Mapping impact kind");
}

llvm::StringRef hardwareMutationFamilySpelling(
    HardwareMutationFamily family) {
  switch (family) {
  case HardwareMutationFamily::SpatialTopology:
    return "spatial_topology";
  case HardwareMutationFamily::InstructionCapacity:
    return "instruction_capacity";
  case HardwareMutationFamily::FuCapability:
    return "fu_capability";
  case HardwareMutationFamily::SpatialMemory:
    return "spatial_memory";
  case HardwareMutationFamily::SpatialFifo:
    return "spatial_fifo";
  case HardwareMutationFamily::TemporalOperandBuffer:
    return "temporal_operand_buffer";
  case HardwareMutationFamily::SpatialSwitch:
    return "spatial_switch";
  case HardwareMutationFamily::SystemAccCore:
    return "system_acc_core";
  case HardwareMutationFamily::SystemInstructionContext:
    return "system_instruction_context";
  case HardwareMutationFamily::SystemTransport:
    return "system_transport";
  case HardwareMutationFamily::SystemMemoryService:
    return "system_memory_service";
  }
  llvm_unreachable("unknown hardware mutation family");
}

llvm::StringRef
hardwareMutationLocalitySpelling(HardwareMutationLocality locality) {
  switch (locality) {
  case HardwareMutationLocality::Unchanged:
    return "unchanged";
  case HardwareMutationLocality::LocalCone:
    return "local_cone";
  case HardwareMutationLocality::GlobalReopen:
    return "global_reopen";
  }
  llvm_unreachable("unknown hardware mutation locality");
}

namespace {

constexpr llvm::StringLiteral topologySchema =
    "loom.spatial_topology_candidate_decision.1.0";
constexpr llvm::StringLiteral microarchitectureSchema =
    "loom.spatial_microarchitecture_candidate_decision.2.1";
constexpr llvm::StringLiteral systemSchema =
    "loom.system_composition_candidate_decision.3.0";

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "hardware_candidate_decision_invalid: " +
                                     message);
}

llvm::Error
validateInstructionStoreResizes(llvm::ArrayRef<ResizeInstructionStore> stores) {
  if (stores.empty())
    return invalid("instruction-store resize set is empty");
  std::vector<std::uint8_t> previous;
  bool first = true;
  for (const ResizeInstructionStore &store : stores) {
    if (store.instructionCapacity == 0)
      return invalid("instruction capacity must be positive");
    const std::vector<std::uint8_t> key =
        loom::fabric::canonicalFabricBytes(store.target);
    if (!first && !(previous < key))
      return invalid(
          "instruction-store resize targets are not canonical and unique");
    first = false;
    previous = key;
  }
  return llvm::Error::success();
}

class Writer final {
public:
  void u8(std::uint8_t value) { bytes_.push_back(value); }
  void u32(std::uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
  }
  void u64(std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
  }
  void framed(llvm::ArrayRef<std::uint8_t> bytes) {
    u64(bytes.size());
    bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
  }
  template <typename Ref> void ref(const Ref &reference) {
    framed(loom::fabric::canonicalFabricBytes(reference));
  }
  void root(const ArtifactRootReference &reference) {
    framed(encodeArtifactRootReference(reference));
  }
  std::vector<std::uint8_t> take() { return std::move(bytes_); }

private:
  std::vector<std::uint8_t> bytes_;
};

class Reader final {
public:
  explicit Reader(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<std::uint8_t> u8() {
    if (bytes_.empty())
      return invalid("truncated u8");
    std::uint8_t result = bytes_.front();
    bytes_ = bytes_.drop_front();
    return result;
  }
  llvm::Expected<std::uint32_t> u32() {
    if (bytes_.size() < 4)
      return invalid("truncated u32");
    std::uint32_t result = 0;
    for (std::uint8_t byte : bytes_.take_front(4))
      result = (result << 8) | byte;
    bytes_ = bytes_.drop_front(4);
    return result;
  }
  llvm::Expected<std::uint64_t> u64() {
    if (bytes_.size() < 8)
      return invalid("truncated u64");
    std::uint64_t result = 0;
    for (std::uint8_t byte : bytes_.take_front(8))
      result = (result << 8) | byte;
    bytes_ = bytes_.drop_front(8);
    return result;
  }
  llvm::Expected<llvm::ArrayRef<std::uint8_t>> framed() {
    auto size = u64();
    if (!size)
      return size.takeError();
    if (*size > bytes_.size())
      return invalid("truncated framed value");
    llvm::ArrayRef<std::uint8_t> result = bytes_.take_front(*size);
    bytes_ = bytes_.drop_front(*size);
    return result;
  }
  template <typename Ref> llvm::Expected<Ref> ref() {
    auto bytes = framed();
    if (!bytes)
      return bytes.takeError();
    return loom::fabric::decodeFabricRef<Ref>(*bytes);
  }
  llvm::Expected<ArtifactRootReference> root() {
    auto bytes = framed();
    if (!bytes)
      return bytes.takeError();
    auto decoded = decodeArtifactRootReferencePrefix(*bytes);
    if (!decoded)
      return decoded.takeError();
    if (decoded->byteCount != bytes->size())
      return invalid("Artifact root reference has trailing bytes");
    return std::move(decoded->reference);
  }
  std::size_t remaining() const { return bytes_.size(); }
  bool empty() const { return bytes_.empty(); }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
};

template <typename Ref>
void writeRefs(Writer &writer, llvm::ArrayRef<Ref> references) {
  writer.u64(references.size());
  for (const Ref &reference : references)
    writer.ref(reference);
}

template <typename Ref>
llvm::Expected<std::vector<Ref>> readRefs(Reader &reader) {
  auto count = reader.u64();
  if (!count)
    return count.takeError();
  if (*count > reader.remaining())
    return invalid("reference count exceeds the remaining payload");
  std::vector<Ref> references;
  references.reserve(*count);
  for (std::uint64_t index = 0; index < *count; ++index) {
    auto reference = reader.ref<Ref>();
    if (!reference)
      return reference.takeError();
    references.push_back(std::move(*reference));
  }
  return references;
}

void writeConnections(
    Writer &writer,
    llvm::ArrayRef<loom::fabric::FabricPointConnectionPayload> connections) {
  writer.u64(connections.size());
  for (const auto &connection : connections) {
    writer.ref(connection.destination);
    writer.ref(connection.source);
  }
}

llvm::Expected<std::vector<loom::fabric::FabricPointConnectionPayload>>
readConnections(Reader &reader) {
  auto count = reader.u64();
  if (!count)
    return count.takeError();
  if (*count > reader.remaining())
    return invalid("connection count exceeds the remaining payload");
  std::vector<loom::fabric::FabricPointConnectionPayload> connections;
  connections.reserve(*count);
  for (std::uint64_t index = 0; index < *count; ++index) {
    auto destination = reader.ref<loom::fabric::FabricTransportEndpointRef>();
    if (!destination)
      return destination.takeError();
    auto source = reader.ref<loom::fabric::FabricTransportEndpointRef>();
    if (!source)
      return source.takeError();
    connections.push_back({std::move(*source), std::move(*destination)});
  }
  return connections;
}

void writeTopologyBody(Writer &writer,
                       const SpatialTopologyDecision &decision) {
  writer.u32(decision.index());
  std::visit(
      [&](const auto &value) {
        using Value = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Value, AddOccurrence>)
          writer.ref(value.prototype);
        else if constexpr (std::is_same_v<Value, RemoveOccurrence>)
          writer.ref(value.target);
        else if constexpr (std::is_same_v<Value, ReplacePointConnection>) {
          writer.ref(value.destination);
          writer.ref(value.source);
        } else if constexpr (std::is_same_v<Value,
                                            AdjustParallelConnectionCount>) {
          writeConnections(writer, value.connections);
        } else {
          writer.u64(value.value.inputCount);
          writeRefs(writer,
                    llvm::ArrayRef<loom::fabric::FabricTransportEndpointRef>(
                        value.value.outputSources));
        }
      },
      decision);
}

llvm::Expected<SpatialTopologyDecision> readTopologyBody(Reader &reader) {
  auto tag = reader.u32();
  if (!tag)
    return tag.takeError();
  switch (*tag) {
  case 0: {
    auto prototype = reader.ref<loom::fabric::FabricModulePhysicalOwnerRef>();
    if (!prototype)
      return prototype.takeError();
    return SpatialTopologyDecision(AddOccurrence{std::move(*prototype)});
  }
  case 1: {
    auto target = reader.ref<loom::fabric::FabricModulePhysicalOwnerRef>();
    if (!target)
      return target.takeError();
    return SpatialTopologyDecision(RemoveOccurrence{std::move(*target)});
  }
  case 2: {
    auto destination = reader.ref<loom::fabric::FabricTransportEndpointRef>();
    if (!destination)
      return destination.takeError();
    auto source = reader.ref<loom::fabric::FabricTransportEndpointRef>();
    if (!source)
      return source.takeError();
    return SpatialTopologyDecision(
        ReplacePointConnection{std::move(*destination), std::move(*source)});
  }
  case 3: {
    auto connections = readConnections(reader);
    if (!connections)
      return connections.takeError();
    if (connections->empty())
      return invalid("parallel connection replacement must be nonempty");
    return SpatialTopologyDecision(
        AdjustParallelConnectionCount{std::move(*connections)});
  }
  case 4: {
    auto inputCount = reader.u64();
    if (!inputCount)
      return inputCount.takeError();
    auto outputs = readRefs<loom::fabric::FabricTransportEndpointRef>(reader);
    if (!outputs)
      return outputs.takeError();
    return SpatialTopologyDecision(ChangeBoundaryInventory{
        BoundaryInventoryValue{*inputCount, std::move(*outputs)}});
  }
  default:
    return invalid("unknown Spatial topology decision tag");
  }
}

void writeMicroarchitectureBody(
    Writer &writer, const SpatialMicroarchitectureDecision &decision) {
  writer.u32(decision.index());
  std::visit(
      [&](const auto &value) {
        using Value = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Value, ResizeInstructionStores>) {
          writer.u64(value.stores.size());
          for (const ResizeInstructionStore &store : value.stores) {
            writer.ref(store.target);
            writer.u32(store.instructionCapacity);
          }
        } else {
          writer.ref(value.target);
          if constexpr (std::is_same_v<Value, ChangePeKind> ||
                        std::is_same_v<Value, ChangeFuCapability> ||
                        std::is_same_v<Value,
                                       ChangeSwitchModeOrScheduleCapacity> ||
                        std::is_same_v<Value, ChangeMemoryOperationTable>) {
            writer.ref(value.prototype);
          } else if constexpr (std::is_same_v<Value, ResizeInstructionStore>) {
            writer.u32(value.instructionCapacity);
          } else if constexpr (std::is_same_v<Value, ChangeFuInventory>) {
            writeRefs(writer,
                      llvm::ArrayRef<loom::fabric::FabricFuOccurrenceRef>(
                          value.prototypes));
          } else if constexpr (std::is_same_v<Value, ResizeMemory>) {
            writer.u64(value.capacityBytes);
          } else if constexpr (std::is_same_v<Value, ResizeFifo>) {
            writer.u32(value.depth);
          } else if constexpr (std::is_same_v<
                                   Value, ChangeTemporalOperandBufferMode>) {
            writer.u32(static_cast<std::uint32_t>(value.mode));
          } else if constexpr (std::is_same_v<
                                   Value, ResizeTemporalOperandBuffer>) {
            writer.u32(value.entriesPerAllocationUnit);
          } else {
            writer.u8(value.bypassable ? 1 : 0);
          }
        }
      },
      decision);
}

llvm::Expected<SpatialMicroarchitectureDecision>
readMicroarchitectureBody(Reader &reader) {
  auto tag = reader.u32();
  if (!tag)
    return tag.takeError();
  switch (*tag) {
  case 0: {
    auto target = reader.ref<loom::fabric::FabricPeOccurrenceRef>();
    if (!target)
      return target.takeError();
    auto prototype = reader.ref<loom::fabric::FabricPeOccurrenceRef>();
    if (!prototype)
      return prototype.takeError();
    return SpatialMicroarchitectureDecision(ChangePeKind{*target, *prototype});
  }
  case 1: {
    auto target = reader.ref<loom::fabric::FabricPeOccurrenceRef>();
    if (!target)
      return target.takeError();
    auto capacity = reader.u32();
    if (!capacity)
      return capacity.takeError();
    if (*capacity == 0)
      return invalid("instruction capacity must be positive");
    return SpatialMicroarchitectureDecision(
        ResizeInstructionStore{*target, *capacity});
  }
  case 2: {
    auto target = reader.ref<loom::fabric::FabricPeOccurrenceRef>();
    if (!target)
      return target.takeError();
    auto prototypes = readRefs<loom::fabric::FabricFuOccurrenceRef>(reader);
    if (!prototypes)
      return prototypes.takeError();
    if (prototypes->empty())
      return invalid("FU inventory must be nonempty");
    return SpatialMicroarchitectureDecision(
        ChangeFuInventory{*target, std::move(*prototypes)});
  }
  case 3: {
    auto target = reader.ref<loom::fabric::FabricFuOccurrenceRef>();
    if (!target)
      return target.takeError();
    auto prototype = reader.ref<loom::fabric::FabricFuOccurrenceRef>();
    if (!prototype)
      return prototype.takeError();
    return SpatialMicroarchitectureDecision(
        ChangeFuCapability{*target, *prototype});
  }
  case 4: {
    auto target = reader.ref<loom::fabric::FabricSwitchOccurrenceRef>();
    if (!target)
      return target.takeError();
    auto prototype = reader.ref<loom::fabric::FabricSwitchOccurrenceRef>();
    if (!prototype)
      return prototype.takeError();
    return SpatialMicroarchitectureDecision(
        ChangeSwitchModeOrScheduleCapacity{*target, *prototype});
  }
  case 5: {
    auto target = reader.ref<loom::fabric::FabricMemoryOccurrenceRef>();
    if (!target)
      return target.takeError();
    auto capacity = reader.u64();
    if (!capacity)
      return capacity.takeError();
    if (*capacity == 0)
      return invalid("memory capacity must be positive");
    return SpatialMicroarchitectureDecision(ResizeMemory{*target, *capacity});
  }
  case 6: {
    auto target = reader.ref<loom::fabric::FabricMemoryOccurrenceRef>();
    if (!target)
      return target.takeError();
    auto prototype = reader.ref<loom::fabric::FabricMemoryOccurrenceRef>();
    if (!prototype)
      return prototype.takeError();
    return SpatialMicroarchitectureDecision(
        ChangeMemoryOperationTable{*target, *prototype});
  }
  case 7: {
    auto target = reader.ref<loom::fabric::FabricFifoOccurrenceRef>();
    if (!target)
      return target.takeError();
    auto depth = reader.u32();
    if (!depth)
      return depth.takeError();
    if (*depth == 0)
      return invalid("FIFO depth must be positive");
    return SpatialMicroarchitectureDecision(ResizeFifo{*target, *depth});
  }
  case 8: {
    auto target = reader.ref<loom::fabric::FabricFifoOccurrenceRef>();
    if (!target)
      return target.takeError();
    auto bypassable = reader.u8();
    if (!bypassable)
      return bypassable.takeError();
    if (*bypassable > 1)
      return invalid("FIFO bypass flag is not canonical");
    return SpatialMicroarchitectureDecision(
        ChangeFifoBypassCapability{*target, *bypassable != 0});
  }
  case 9: {
    auto count = reader.u64();
    if (!count)
      return count.takeError();
    if (*count == 0 || *count > reader.remaining())
      return invalid("instruction-store resize count is invalid");
    std::vector<ResizeInstructionStore> stores;
    stores.reserve(*count);
    for (std::uint64_t ordinal = 0; ordinal != *count; ++ordinal) {
      auto target = reader.ref<loom::fabric::FabricPeOccurrenceRef>();
      if (!target)
        return target.takeError();
      auto capacity = reader.u32();
      if (!capacity)
        return capacity.takeError();
      stores.push_back({*target, *capacity});
    }
    if (llvm::Error error = validateInstructionStoreResizes(stores))
      return std::move(error);
    return SpatialMicroarchitectureDecision(
        ResizeInstructionStores{std::move(stores)});
  }
  case 10: {
    auto target = reader.ref<loom::fabric::FabricPeOccurrenceRef>();
    if (!target)
      return target.takeError();
    auto mode = reader.u32();
    if (!mode)
      return mode.takeError();
    auto value = ::fabric::symbolizeOperandBufferMode(*mode);
    if (!value)
      return invalid("operand-buffer mode is outside its closed domain");
    return SpatialMicroarchitectureDecision(
        ChangeTemporalOperandBufferMode{*target, *value});
  }
  case 11: {
    auto target = reader.ref<loom::fabric::FabricPeOccurrenceRef>();
    if (!target)
      return target.takeError();
    auto entries = reader.u32();
    if (!entries)
      return entries.takeError();
    if (*entries == 0)
      return invalid("operand-buffer entries must be positive");
    return SpatialMicroarchitectureDecision(
        ResizeTemporalOperandBuffer{*target, *entries});
  }
  default:
    return invalid("unknown Spatial microarchitecture decision tag");
  }
}

void writeSystemBody(Writer &writer,
                     const SystemCompositionDecision &decision) {
  writer.u32(decision.index());
  std::visit(
      [&](const auto &value) {
        using Value = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Value, AddAccCore>) {
          writer.ref(value.prototype);
          writer.root(value.module);
        } else if constexpr (std::is_same_v<Value, RemoveAccCore>) {
          writer.ref(value.target);
        } else if constexpr (std::is_same_v<Value, ReplaceSpatialAttachment>) {
          writer.ref(value.target);
          writer.root(value.module);
        } else if constexpr (std::is_same_v<Value,
                                            SelectInstructionCoreRealization> ||
                             std::is_same_v<Value, ChangeTransportResource>) {
          writer.ref(value.target);
          writer.ref(value.prototype);
        } else if constexpr (std::is_same_v<Value, ChangeTransportConnection>) {
          writer.ref(value.destination);
          writer.ref(value.source);
        } else {
          writer.u32(value.value.index());
          std::visit(
              [&](const auto &attachment) {
                using Attachment = std::decay_t<decltype(attachment)>;
                if constexpr (std::is_same_v<Attachment,
                                             ChangeSpatialMemoryAttachment>) {
                  writer.ref(attachment.spatialEndpoint);
                  writer.ref(attachment.serviceEndpoint);
                } else {
                  writer.ref(attachment.destination);
                  writer.ref(attachment.source);
                }
              },
              value.value);
        }
      },
      decision);
}

llvm::Expected<SystemCompositionDecision> readSystemBody(Reader &reader) {
  auto tag = reader.u32();
  if (!tag)
    return tag.takeError();
  switch (*tag) {
  case 0: {
    auto prototype = reader.ref<loom::fabric::AccCoreOccurrenceRef>();
    if (!prototype)
      return prototype.takeError();
    auto module = reader.root();
    if (!module)
      return module.takeError();
    return SystemCompositionDecision(AddAccCore{*prototype, *module});
  }
  case 1: {
    auto target = reader.ref<loom::fabric::AccCoreOccurrenceRef>();
    if (!target)
      return target.takeError();
    return SystemCompositionDecision(RemoveAccCore{*target});
  }
  case 2: {
    auto target = reader.ref<loom::fabric::AccCoreOccurrenceRef>();
    if (!target)
      return target.takeError();
    auto module = reader.root();
    if (!module)
      return module.takeError();
    return SystemCompositionDecision(
        ReplaceSpatialAttachment{*target, *module});
  }
  case 3: {
    auto target = reader.ref<loom::fabric::InstructionCoreContextRef>();
    if (!target)
      return target.takeError();
    auto prototype = reader.ref<loom::fabric::InstructionCoreContextRef>();
    if (!prototype)
      return prototype.takeError();
    return SystemCompositionDecision(
        SelectInstructionCoreRealization{*target, *prototype});
  }
  case 4: {
    auto target = reader.ref<loom::fabric::SystemTransportResourceRef>();
    if (!target)
      return target.takeError();
    auto prototype = reader.ref<loom::fabric::SystemTransportResourceRef>();
    if (!prototype)
      return prototype.takeError();
    return SystemCompositionDecision(
        ChangeTransportResource{*target, *prototype});
  }
  case 5: {
    auto destination = reader.ref<loom::fabric::FabricTransportEndpointRef>();
    if (!destination)
      return destination.takeError();
    auto source = reader.ref<loom::fabric::FabricTransportEndpointRef>();
    if (!source)
      return source.takeError();
    return SystemCompositionDecision(
        ChangeTransportConnection{*destination, *source});
  }
  case 6: {
    auto attachmentTag = reader.u32();
    if (!attachmentTag)
      return attachmentTag.takeError();
    auto destination = reader.ref<loom::fabric::FabricMemoryEndpointRef>();
    if (!destination)
      return destination.takeError();
    if (*attachmentTag == 0) {
      auto service = reader.ref<loom::fabric::SystemServiceEndpointRef>();
      if (!service)
        return service.takeError();
      return SystemCompositionDecision(ChangeServiceOrMemoryAttachment{
          ChangeSpatialMemoryAttachment{*destination, *service}});
    }
    if (*attachmentTag == 1) {
      auto source = reader.ref<loom::fabric::FabricMemoryEndpointRef>();
      if (!source)
        return source.takeError();
      return SystemCompositionDecision(ChangeServiceOrMemoryAttachment{
          ChangeMemoryServiceConnection{*destination, *source}});
    }
    return invalid("unknown service or memory attachment decision tag");
  }
  default:
    return invalid("unknown System composition decision tag");
  }
}

llvm::Error validateEntityCorrespondence(
    llvm::ArrayRef<loom::fabric::FabricSystemEntityCorrespondence>
        correspondence) {
  std::optional<std::pair<loom::fabric::FabricEntityKind,
                          loom::fabric::FabricEntityId>>
      previousSource;
  std::set<std::pair<loom::fabric::FabricEntityKind,
                     loom::fabric::FabricEntityId>>
      targetKeys;
  for (const auto &entry : correspondence) {
    const auto source = std::make_pair(entry.source.kind, entry.source.id);
    const auto target = std::make_pair(entry.target.kind, entry.target.id);
    if (entry.source.kind != entry.target.kind)
      return invalid("System entity lineage changes an entity kind");
    if (previousSource && !(*previousSource < source))
      return invalid("System entity lineage sources are not canonical");
    previousSource = source;
    if (!targetKeys.insert(target).second)
      return invalid("System entity lineage maps two sources to one target");
  }
  return llvm::Error::success();
}

void writeEntityCorrespondence(
    Writer &writer,
    llvm::ArrayRef<loom::fabric::FabricSystemEntityCorrespondence>
        correspondence) {
  writer.u64(correspondence.size());
  for (const auto &entry : correspondence) {
    writer.u32(static_cast<std::uint32_t>(entry.source.kind));
    writer.u64(entry.source.id);
    writer.u64(entry.target.id);
  }
}

llvm::Expected<
    std::vector<loom::fabric::FabricSystemEntityCorrespondence>>
readEntityCorrespondence(Reader &reader) {
  auto count = reader.u64();
  if (!count)
    return count.takeError();
  if (*count > reader.remaining())
    return invalid("System entity lineage count exceeds its payload");
  std::vector<loom::fabric::FabricSystemEntityCorrespondence> result;
  result.reserve(*count);
  const std::uint32_t kindCount = loom::fabric::fabricClosedBound(
      loom::fabric::FabricEntityKind());
  for (std::uint64_t ordinal = 0; ordinal != *count; ++ordinal) {
    auto rawKind = reader.u32();
    if (!rawKind)
      return rawKind.takeError();
    if (*rawKind >= kindCount)
      return invalid("System entity lineage has an unknown entity kind");
    auto source = reader.u64();
    if (!source)
      return source.takeError();
    auto target = reader.u64();
    if (!target)
      return target.takeError();
    const auto kind =
        static_cast<loom::fabric::FabricEntityKind>(*rawKind);
    result.push_back({{kind, *source}, {kind, *target}});
  }
  if (llvm::Error error = validateEntityCorrespondence(result))
    return std::move(error);
  return result;
}

llvm::Error validateTransferPatternCorrespondence(
    llvm::ArrayRef<loom::fabric::FabricSystemTransferPatternCorrespondence>
        correspondence) {
  std::vector<std::uint8_t> previousSource;
  std::set<std::vector<std::uint8_t>> targetKeys;
  bool first = true;
  for (const auto &entry : correspondence) {
    std::vector<std::uint8_t> source =
        loom::fabric::canonicalFabricBytes(entry.source);
    if (!first && !(previousSource < source))
      return invalid("System transfer-pattern lineage is not canonical");
    first = false;
    previousSource = std::move(source);
    if (!targetKeys.insert(
                       loom::fabric::canonicalFabricBytes(entry.target))
             .second)
      return invalid(
          "System transfer-pattern lineage maps two sources to one target");
  }
  return llvm::Error::success();
}

void writeTransferPatternCorrespondence(
    Writer &writer,
    llvm::ArrayRef<loom::fabric::FabricSystemTransferPatternCorrespondence>
        correspondence) {
  writer.u64(correspondence.size());
  for (const auto &entry : correspondence) {
    writer.ref(entry.source);
    writer.ref(entry.target);
  }
}

llvm::Expected<
    std::vector<loom::fabric::FabricSystemTransferPatternCorrespondence>>
readTransferPatternCorrespondence(Reader &reader) {
  auto count = reader.u64();
  if (!count)
    return count.takeError();
  if (*count > reader.remaining())
    return invalid(
        "System transfer-pattern lineage count exceeds its payload");
  std::vector<loom::fabric::FabricSystemTransferPatternCorrespondence> result;
  result.reserve(*count);
  for (std::uint64_t ordinal = 0; ordinal != *count; ++ordinal) {
    auto source = reader.ref<loom::fabric::FabricTransferPatternRef>();
    if (!source)
      return source.takeError();
    auto target = reader.ref<loom::fabric::FabricTransferPatternRef>();
    if (!target)
      return target.takeError();
    result.push_back({*source, *target});
  }
  if (llvm::Error error = validateTransferPatternCorrespondence(result))
    return std::move(error);
  return result;
}

template <typename Decision, typename WriteBody>
std::vector<std::uint8_t> encodeDecision(const ArtifactRootReference &parent,
                                         const Decision &decision,
                                         WriteBody writeBody) {
  Writer writer;
  writer.u32(1);
  writer.root(parent);
  writeBody(writer, decision);
  return writer.take();
}

template <typename Record, typename ReadBody>
llvm::Expected<Record> adoptDecision(llvm::ArrayRef<std::uint8_t> bytes,
                                     ReadBody readBody) {
  Reader reader(bytes);
  auto version = reader.u32();
  if (!version)
    return version.takeError();
  if (*version != 1)
    return invalid("unsupported candidate decision version");
  auto parent = reader.root();
  if (!parent)
    return parent.takeError();
  auto decision = readBody(reader);
  if (!decision)
    return decision.takeError();
  if (!reader.empty())
    return invalid("candidate decision has trailing bytes");
  Record record{std::move(*parent), std::move(*decision)};
  return record;
}

template <typename Decision, typename WriteBody>
llvm::Expected<std::vector<Decision>>
canonicalizeDecisions(std::vector<Decision> decisions, WriteBody writeBody) {
  std::vector<std::pair<std::vector<std::uint8_t>, Decision>> keyed;
  keyed.reserve(decisions.size());
  for (Decision &decision : decisions) {
    Writer writer;
    writeBody(writer, decision);
    keyed.emplace_back(writer.take(), std::move(decision));
  }
  llvm::sort(keyed, [](const auto &left, const auto &right) {
    return left.first < right.first;
  });
  for (std::size_t index = 1; index < keyed.size(); ++index)
    if (keyed[index - 1].first == keyed[index].first)
      return invalid("decision domains contain a duplicate value");
  std::vector<Decision> result;
  result.reserve(keyed.size());
  for (auto &entry : keyed)
    result.push_back(std::move(entry.second));
  return result;
}

template <typename Decision, typename WriteBody>
std::vector<std::uint8_t>
encodeRewriteConfig(llvm::ArrayRef<Decision> decisions,
                    std::uint64_t maxChildrenPerParent, WriteBody writeBody) {
  Writer writer;
  writer.u32(1);
  writer.u64(maxChildrenPerParent);
  writer.u64(decisions.size());
  for (const Decision &decision : decisions) {
    Writer decisionWriter;
    writeBody(decisionWriter, decision);
    writer.framed(decisionWriter.take());
  }
  return writer.take();
}

template <typename Decision, typename ReadBody, typename WriteBody>
llvm::Expected<std::pair<std::vector<Decision>, std::uint64_t>>
adoptRewriteConfig(llvm::ArrayRef<std::uint8_t> bytes, ReadBody readBody,
                   WriteBody writeBody) {
  Reader reader(bytes);
  auto version = reader.u32();
  if (!version)
    return version.takeError();
  if (*version != 1)
    return invalid("unsupported rewrite config version");
  auto maxChildren = reader.u64();
  if (!maxChildren)
    return maxChildren.takeError();
  if (*maxChildren == 0)
    return invalid("max children per parent must be positive");
  auto count = reader.u64();
  if (!count)
    return count.takeError();
  if (*count == 0)
    return invalid("rewrite config has no decisions");
  if (*count > reader.remaining())
    return invalid("decision count exceeds the remaining config payload");
  std::vector<Decision> decisions;
  decisions.reserve(*count);
  for (std::uint64_t index = 0; index < *count; ++index) {
    auto body = reader.framed();
    if (!body)
      return body.takeError();
    Reader decisionReader(*body);
    auto decision = readBody(decisionReader);
    if (!decision)
      return decision.takeError();
    if (!decisionReader.empty())
      return invalid("rewrite decision has trailing bytes");
    decisions.push_back(std::move(*decision));
  }
  if (!reader.empty())
    return invalid("rewrite config has trailing bytes");
  auto canonical = canonicalizeDecisions(std::move(decisions), writeBody);
  if (!canonical)
    return canonical.takeError();
  std::vector<std::uint8_t> reencoded =
      encodeRewriteConfig<Decision>(*canonical, *maxChildren, writeBody);
  if (llvm::ArrayRef<std::uint8_t>(reencoded) != bytes)
    return invalid("rewrite config is not in canonical decision order");
  return std::make_pair(std::move(*canonical), *maxChildren);
}

template <typename Domain, typename KeyWriter>
llvm::Error admitDomainKeys(llvm::ArrayRef<Domain> domains,
                            KeyWriter writeKey) {
  if (domains.empty())
    return invalid("decision-domain set must be nonempty");
  std::set<std::vector<std::uint8_t>> keys;
  for (const Domain &domain : domains) {
    Writer writer;
    writer.u32(domain.index());
    writeKey(writer, domain);
    if (!keys.insert(writer.take()).second)
      return invalid("decision-domain set repeats a canonical domain key");
  }
  return llvm::Error::success();
}

template <typename Values>
llvm::Error requireValues(const Values &values, llvm::StringRef description) {
  if (values.empty())
    return invalid(description + " domain has no finite values");
  return llvm::Error::success();
}

} // namespace

llvm::Expected<std::vector<SpatialTopologyDecision>>
expandSpatialTopologyDecisionDomains(
    llvm::ArrayRef<SpatialTopologyDecisionDomain> domains) {
  if (llvm::Error error = admitDomainKeys(domains, [](Writer &writer,
                                                      const auto &domain) {
        std::visit(
            [&](const auto &value) {
              using Value = std::decay_t<decltype(value)>;
              if constexpr (std::is_same_v<Value, ReplacePointConnectionDomain>)
                writer.ref(value.destination);
            },
            domain);
      }))
    return std::move(error);
  std::vector<SpatialTopologyDecision> decisions;
  for (const auto &domain : domains) {
    llvm::Error validation = std::visit(
        [](const auto &value) {
          using Value = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<Value, AddOccurrenceDomain>)
            return requireValues(value.prototypes, "Spatial topology");
          else if constexpr (std::is_same_v<Value, RemoveOccurrenceDomain>)
            return requireValues(value.targets, "Spatial topology");
          else if constexpr (std::is_same_v<Value,
                                            ReplacePointConnectionDomain>)
            return requireValues(value.sources, "Spatial topology");
          else
            return requireValues(value.values, "Spatial topology");
        },
        domain);
    if (validation)
      return std::move(validation);
    std::visit(
        [&](const auto &value) {
          using Value = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<Value, AddOccurrenceDomain>)
            for (const auto &prototype : value.prototypes)
              decisions.push_back(AddOccurrence{prototype});
          else if constexpr (std::is_same_v<Value, RemoveOccurrenceDomain>)
            for (const auto &target : value.targets)
              decisions.push_back(RemoveOccurrence{target});
          else if constexpr (std::is_same_v<Value,
                                            ReplacePointConnectionDomain>)
            for (const auto &source : value.sources)
              decisions.push_back(
                  ReplacePointConnection{value.destination, source});
          else if constexpr (std::is_same_v<
                                 Value, AdjustParallelConnectionCountDomain>)
            for (const auto &connections : value.values)
              decisions.push_back(AdjustParallelConnectionCount{connections});
          else
            for (const auto &boundary : value.values)
              decisions.push_back(ChangeBoundaryInventory{boundary});
        },
        domain);
  }
  return canonicalizeDecisions(std::move(decisions), writeTopologyBody);
}

llvm::Expected<std::vector<SpatialMicroarchitectureDecision>>
expandSpatialMicroarchitectureDecisionDomains(
    llvm::ArrayRef<SpatialMicroarchitectureDecisionDomain> domains) {
  if (llvm::Error error =
          admitDomainKeys(domains, [](Writer &writer, const auto &domain) {
            std::visit(
                [&](const auto &value) {
                  using Value = std::decay_t<decltype(value)>;
                  if constexpr (std::is_same_v<Value,
                                               ResizeInstructionStoresDomain>) {
                    for (const ResizeInstructionStore &store : value.stores) {
                      writer.ref(store.target);
                      writer.u32(store.instructionCapacity);
                    }
                  } else {
                    writer.ref(value.target);
                  }
                },
                domain);
          }))
    return std::move(error);
  std::vector<SpatialMicroarchitectureDecision> decisions;
  for (const auto &domain : domains) {
    llvm::Error validation = std::visit(
        [](const auto &value) {
          using Value = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<Value, ChangePeKindDomain> ||
                        std::is_same_v<Value, ChangeFuCapabilityDomain> ||
                        std::is_same_v<
                            Value, ChangeSwitchModeOrScheduleCapacityDomain> ||
                        std::is_same_v<Value, ChangeMemoryOperationTableDomain>)
            return requireValues(value.prototypes, "microarchitecture");
          else if constexpr (std::is_same_v<Value,
                                            ResizeInstructionStoreDomain>)
            return requireValues(value.capacities, "microarchitecture");
          else if constexpr (std::is_same_v<Value,
                                            ResizeInstructionStoresDomain>)
            return validateInstructionStoreResizes(value.stores);
          else if constexpr (std::is_same_v<Value, ResizeMemoryDomain>)
            return requireValues(value.capacitiesBytes, "microarchitecture");
          else if constexpr (std::is_same_v<Value, ResizeFifoDomain>)
            return requireValues(value.depths, "microarchitecture");
          else if constexpr (std::is_same_v<
                                 Value,
                                 ChangeTemporalOperandBufferModeDomain>)
            return requireValues(value.modes, "microarchitecture");
          else if constexpr (std::is_same_v<
                                 Value, ResizeTemporalOperandBufferDomain>)
            return requireValues(value.entriesPerAllocationUnit,
                                 "microarchitecture");
          else
            return requireValues(value.values, "microarchitecture");
        },
        domain);
    if (validation)
      return std::move(validation);
    std::visit(
        [&](const auto &value) {
          using Value = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<Value, ChangePeKindDomain>)
            for (auto prototype : value.prototypes)
              decisions.push_back(ChangePeKind{value.target, prototype});
          else if constexpr (std::is_same_v<Value,
                                            ResizeInstructionStoreDomain>)
            for (auto capacity : value.capacities)
              decisions.push_back(
                  ResizeInstructionStore{value.target, capacity});
          else if constexpr (std::is_same_v<Value,
                                            ResizeInstructionStoresDomain>)
            decisions.push_back(ResizeInstructionStores{value.stores});
          else if constexpr (std::is_same_v<Value, ChangeFuInventoryDomain>)
            for (const auto &prototypes : value.values)
              decisions.push_back(ChangeFuInventory{value.target, prototypes});
          else if constexpr (std::is_same_v<Value, ChangeFuCapabilityDomain>)
            for (auto prototype : value.prototypes)
              decisions.push_back(ChangeFuCapability{value.target, prototype});
          else if constexpr (std::is_same_v<
                                 Value,
                                 ChangeSwitchModeOrScheduleCapacityDomain>)
            for (auto prototype : value.prototypes)
              decisions.push_back(
                  ChangeSwitchModeOrScheduleCapacity{value.target, prototype});
          else if constexpr (std::is_same_v<Value, ResizeMemoryDomain>)
            for (auto capacity : value.capacitiesBytes)
              decisions.push_back(ResizeMemory{value.target, capacity});
          else if constexpr (std::is_same_v<Value,
                                            ChangeMemoryOperationTableDomain>)
            for (auto prototype : value.prototypes)
              decisions.push_back(
                  ChangeMemoryOperationTable{value.target, prototype});
          else if constexpr (std::is_same_v<Value, ResizeFifoDomain>)
            for (auto depth : value.depths)
              decisions.push_back(ResizeFifo{value.target, depth});
          else if constexpr (std::is_same_v<
                                 Value,
                                 ChangeTemporalOperandBufferModeDomain>)
            for (auto mode : value.modes)
              decisions.push_back(
                  ChangeTemporalOperandBufferMode{value.target, mode});
          else if constexpr (std::is_same_v<
                                 Value, ResizeTemporalOperandBufferDomain>)
            for (auto entries : value.entriesPerAllocationUnit)
              decisions.push_back(
                  ResizeTemporalOperandBuffer{value.target, entries});
          else
            for (bool bypassable : value.values)
              decisions.push_back(
                  ChangeFifoBypassCapability{value.target, bypassable});
        },
        domain);
  }
  return canonicalizeDecisions(std::move(decisions),
                               writeMicroarchitectureBody);
}

llvm::Expected<std::vector<SystemCompositionDecision>>
expandSystemCompositionDecisionDomains(
    llvm::ArrayRef<SystemCompositionDecisionDomain> domains) {
  if (llvm::Error error = admitDomainKeys(domains, [](Writer &writer,
                                                      const auto &domain) {
        std::visit(
            [&](const auto &value) {
              using Value = std::decay_t<decltype(value)>;
              if constexpr (std::is_same_v<Value, AddAccCoreDomain>)
                writer.ref(value.prototype);
              else if constexpr (std::is_same_v<Value, RemoveAccCoreDomain>) {
              } else if constexpr (std::is_same_v<
                                       Value,
                                       ChangeServiceOrMemoryAttachmentDomain>) {
                std::visit(
                    [&](const auto &attachment) {
                      using Attachment = std::decay_t<decltype(attachment)>;
                      if constexpr (std::is_same_v<
                                        Attachment,
                                        ChangeSpatialMemoryAttachmentDomain>)
                        writer.ref(attachment.spatialEndpoint);
                      else
                        writer.ref(attachment.destination);
                    },
                    value);
              } else if constexpr (
                  std::is_same_v<Value, ReplaceSpatialAttachmentDomain> ||
                  std::is_same_v<Value,
                                 SelectInstructionCoreRealizationDomain> ||
                  std::is_same_v<Value, ChangeTransportResourceDomain>)
                writer.ref(value.target);
              else
                writer.ref(value.destination);
            },
            domain);
      }))
    return std::move(error);
  std::vector<SystemCompositionDecision> decisions;
  for (const auto &domain : domains) {
    llvm::Error validation = std::visit(
        [](const auto &value) -> llvm::Error {
          using Value = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<Value,
                                       ChangeServiceOrMemoryAttachmentDomain>)
            return std::visit(
                [](const auto &attachment) {
                  using Attachment = std::decay_t<decltype(attachment)>;
                  if constexpr (std::is_same_v<
                                    Attachment,
                                    ChangeSpatialMemoryAttachmentDomain>)
                    return requireValues(attachment.serviceEndpoints,
                                         "service attachment");
                  else
                    return requireValues(attachment.sources,
                                         "memory connection");
                },
                value);
          else if constexpr (std::is_same_v<Value, AddAccCoreDomain> ||
                             std::is_same_v<Value,
                                            ReplaceSpatialAttachmentDomain>)
            return requireValues(value.modules, "System composition");
          else if constexpr (std::is_same_v<Value, RemoveAccCoreDomain>)
            return requireValues(value.targets, "System composition");
          else if constexpr (std::is_same_v<
                                 Value,
                                 SelectInstructionCoreRealizationDomain> ||
                             std::is_same_v<Value,
                                            ChangeTransportResourceDomain>)
            return requireValues(value.prototypes, "System composition");
          else
            return requireValues(value.sources, "System composition");
        },
        domain);
    if (validation)
      return std::move(validation);
    std::visit(
        [&](const auto &value) {
          using Value = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<Value, AddAccCoreDomain>)
            for (const auto &module : value.modules)
              decisions.push_back(AddAccCore{value.prototype, module});
          else if constexpr (std::is_same_v<Value, RemoveAccCoreDomain>)
            for (auto target : value.targets)
              decisions.push_back(RemoveAccCore{target});
          else if constexpr (std::is_same_v<Value,
                                            ReplaceSpatialAttachmentDomain>)
            for (const auto &module : value.modules)
              decisions.push_back(
                  ReplaceSpatialAttachment{value.target, module});
          else if constexpr (std::is_same_v<
                                 Value, SelectInstructionCoreRealizationDomain>)
            for (auto prototype : value.prototypes)
              decisions.push_back(
                  SelectInstructionCoreRealization{value.target, prototype});
          else if constexpr (std::is_same_v<Value,
                                            ChangeTransportResourceDomain>)
            for (auto prototype : value.prototypes)
              decisions.push_back(
                  ChangeTransportResource{value.target, prototype});
          else if constexpr (std::is_same_v<Value,
                                            ChangeTransportConnectionDomain>)
            for (const auto &source : value.sources)
              decisions.push_back(
                  ChangeTransportConnection{value.destination, source});
          else
            std::visit(
                [&](const auto &attachment) {
                  using Attachment = std::decay_t<decltype(attachment)>;
                  if constexpr (std::is_same_v<
                                    Attachment,
                                    ChangeSpatialMemoryAttachmentDomain>)
                    for (auto endpoint : attachment.serviceEndpoints)
                      decisions.push_back(ChangeServiceOrMemoryAttachment{
                          ChangeSpatialMemoryAttachment{
                              attachment.spatialEndpoint, endpoint}});
                  else
                    for (const auto &source : attachment.sources)
                      decisions.push_back(ChangeServiceOrMemoryAttachment{
                          ChangeMemoryServiceConnection{attachment.destination,
                                                        source}});
                },
                value);
        },
        domain);
  }
  return canonicalizeDecisions(std::move(decisions), writeSystemBody);
}

std::vector<std::uint8_t> encodeSpatialTopologyRewriteConfig(
    llvm::ArrayRef<SpatialTopologyDecision> decisions,
    std::uint64_t maxChildrenPerParent) {
  return encodeRewriteConfig(decisions, maxChildrenPerParent,
                             writeTopologyBody);
}

std::vector<std::uint8_t> encodeSpatialMicroarchitectureRewriteConfig(
    llvm::ArrayRef<SpatialMicroarchitectureDecision> decisions,
    std::uint64_t maxChildrenPerParent) {
  return encodeRewriteConfig(decisions, maxChildrenPerParent,
                             writeMicroarchitectureBody);
}

std::vector<std::uint8_t> encodeSystemCompositionRewriteConfig(
    llvm::ArrayRef<SystemCompositionDecision> decisions,
    std::uint64_t maxChildrenPerParent) {
  return encodeRewriteConfig(decisions, maxChildrenPerParent, writeSystemBody);
}

llvm::Expected<std::pair<std::vector<SpatialTopologyDecision>, std::uint64_t>>
adoptSpatialTopologyRewriteConfig(llvm::ArrayRef<std::uint8_t> bytes) {
  return adoptRewriteConfig<SpatialTopologyDecision>(bytes, readTopologyBody,
                                                     writeTopologyBody);
}

llvm::Expected<
    std::pair<std::vector<SpatialMicroarchitectureDecision>, std::uint64_t>>
adoptSpatialMicroarchitectureRewriteConfig(llvm::ArrayRef<std::uint8_t> bytes) {
  return adoptRewriteConfig<SpatialMicroarchitectureDecision>(
      bytes, readMicroarchitectureBody, writeMicroarchitectureBody);
}

llvm::Expected<std::pair<std::vector<SystemCompositionDecision>, std::uint64_t>>
adoptSystemCompositionRewriteConfig(llvm::ArrayRef<std::uint8_t> bytes) {
  return adoptRewriteConfig<SystemCompositionDecision>(bytes, readSystemBody,
                                                       writeSystemBody);
}

std::vector<std::uint8_t>
encodeSpatialTopologyDecision(const ArtifactRootReference &parent,
                              const SpatialTopologyDecision &decision) {
  return encodeDecision(parent, decision, writeTopologyBody);
}

std::vector<std::uint8_t> encodeSpatialMicroarchitectureDecision(
    const ArtifactRootReference &parent,
    const SpatialMicroarchitectureDecision &decision) {
  return encodeDecision(parent, decision, writeMicroarchitectureBody);
}

std::vector<std::uint8_t> encodeSystemCompositionDecision(
    const ArtifactRootReference &parent,
    const SystemCompositionDecision &decision,
    llvm::ArrayRef<loom::fabric::FabricSystemEntityCorrespondence> entities,
    llvm::ArrayRef<loom::fabric::FabricSystemTransferPatternCorrespondence>
        transferPatterns) {
  Writer writer;
  writer.u32(3);
  writer.root(parent);
  writeSystemBody(writer, decision);
  writeEntityCorrespondence(writer, entities);
  writeTransferPatternCorrespondence(writer, transferPatterns);
  return writer.take();
}

llvm::Expected<SpatialTopologyCandidateDecision>
adoptSpatialTopologyDecision(llvm::ArrayRef<std::uint8_t> bytes) {
  return adoptDecision<SpatialTopologyCandidateDecision>(bytes,
                                                         readTopologyBody);
}

llvm::Expected<SpatialMicroarchitectureCandidateDecision>
adoptSpatialMicroarchitectureDecision(llvm::ArrayRef<std::uint8_t> bytes) {
  return adoptDecision<SpatialMicroarchitectureCandidateDecision>(
      bytes, readMicroarchitectureBody);
}

llvm::Expected<SystemCompositionCandidateDecision>
adoptSystemCompositionDecision(llvm::ArrayRef<std::uint8_t> bytes) {
  Reader reader(bytes);
  auto version = reader.u32();
  if (!version)
    return version.takeError();
  if (*version != 3)
    return invalid("unsupported System candidate decision version");
  auto parent = reader.root();
  if (!parent)
    return parent.takeError();
  auto decision = readSystemBody(reader);
  if (!decision)
    return decision.takeError();
  auto entities = readEntityCorrespondence(reader);
  if (!entities)
    return entities.takeError();
  auto transferPatterns = readTransferPatternCorrespondence(reader);
  if (!transferPatterns)
    return transferPatterns.takeError();
  if (!reader.empty())
    return invalid("System candidate decision has trailing bytes");
  return SystemCompositionCandidateDecision{
      std::move(*parent), std::move(*decision), std::move(*entities),
      std::move(*transferPatterns)};
}

llvm::ArrayRef<std::uint8_t> spatialTopologyDecisionSchemaBytes() {
  return {reinterpret_cast<const std::uint8_t *>(topologySchema.data()),
          topologySchema.size()};
}

llvm::ArrayRef<std::uint8_t> spatialMicroarchitectureDecisionSchemaBytes() {
  return {
      reinterpret_cast<const std::uint8_t *>(microarchitectureSchema.data()),
      microarchitectureSchema.size()};
}

llvm::ArrayRef<std::uint8_t> systemCompositionDecisionSchemaBytes() {
  return {reinterpret_cast<const std::uint8_t *>(systemSchema.data()),
          systemSchema.size()};
}

namespace {

template <typename Ref>
void canonicalizeImpactRoots(std::vector<Ref> &roots) {
  llvm::sort(roots, [](const Ref &lhs, const Ref &rhs) {
    return loom::fabric::canonicalFabricBytes(lhs) <
           loom::fabric::canonicalFabricBytes(rhs);
  });
  roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
}

void canonicalizeImpact(HardwareImpactProjection &impact) {
  canonicalizeImpactRoots(impact.tech.realizationRoots);
  canonicalizeImpactRoots(impact.spatial.placementRoots);
  canonicalizeImpactRoots(impact.spatial.routeRoots);
  canonicalizeImpactRoots(impact.system.executionRoots);
  canonicalizeImpactRoots(impact.system.instructionContextRoots);
  canonicalizeImpactRoots(impact.system.transportRoots);
  canonicalizeImpactRoots(impact.system.routeRoots);
  canonicalizeImpactRoots(impact.system.serviceRoots);
  canonicalizeImpactRoots(impact.system.memoryRoots);
}

template <typename Ref>
void addModuleRoot(std::vector<loom::fabric::FabricModulePhysicalOwnerRef> &out,
                   const Ref &reference) {
  out.push_back(llvm::cantFail(
      loom::fabric::FabricModulePhysicalOwnerRef::create(reference)));
}

} // namespace

HardwareImpactProjection projectHardwareImpact(
    const SpatialTopologyCandidateDecision &candidate,
    std::optional<ArtifactRootReference> child) {
  HardwareImpactProjection impact{candidate.parent, std::move(child), {}, {},
                                  {}};
  impact.tech.kind = HardwareMappingImpactKind::Rebase;
  impact.spatial.kind = HardwareMappingImpactKind::Rebase;
  impact.system.kind = HardwareMappingImpactKind::Rebase;
  impact.family = HardwareMutationFamily::SpatialTopology;
  impact.locality = HardwareMutationLocality::LocalCone;
  std::visit(
      [&](const auto &decision) {
        using Decision = std::decay_t<decltype(decision)>;
        if constexpr (std::is_same_v<Decision, AddOccurrence>) {
          return;
        } else if constexpr (std::is_same_v<Decision, RemoveOccurrence>) {
          impact.tech.kind = HardwareMappingImpactKind::Reopen;
          impact.spatial.kind = HardwareMappingImpactKind::Reopen;
          impact.tech.realizationRoots.push_back(decision.target);
          impact.spatial.placementRoots.push_back(decision.target);
        } else if constexpr (std::is_same_v<Decision,
                                            ReplacePointConnection>) {
          impact.locality = HardwareMutationLocality::GlobalReopen;
          impact.spatial.kind = HardwareMappingImpactKind::Reopen;
          impact.spatial.routeRoots.push_back(decision.destination);
          impact.spatial.routeRoots.push_back(decision.source);
        } else if constexpr (std::is_same_v<Decision,
                                            AdjustParallelConnectionCount>) {
          impact.locality = HardwareMutationLocality::GlobalReopen;
          impact.spatial.kind = HardwareMappingImpactKind::Reopen;
          for (const auto &connection : decision.connections) {
            impact.spatial.routeRoots.push_back(connection.destination);
            impact.spatial.routeRoots.push_back(connection.source);
          }
        } else {
          impact.locality = HardwareMutationLocality::GlobalReopen;
          impact.spatial.kind = HardwareMappingImpactKind::Reopen;
          impact.spatial.routeRoots = decision.value.outputSources;
        }
      },
      candidate.decision);
  canonicalizeImpact(impact);
  return impact;
}

HardwareImpactProjection projectHardwareImpact(
    const SpatialMicroarchitectureCandidateDecision &candidate,
    std::optional<ArtifactRootReference> child) {
  HardwareImpactProjection impact{candidate.parent, std::move(child), {}, {},
                                  {}};
  impact.tech.kind = HardwareMappingImpactKind::Rebase;
  impact.spatial.kind = HardwareMappingImpactKind::Rebase;
  impact.system.kind = HardwareMappingImpactKind::Rebase;
  std::visit(
      [&](const auto &decision) {
        using Decision = std::decay_t<decltype(decision)>;
        if constexpr (std::is_same_v<Decision, ResizeInstructionStores>) {
          impact.family = HardwareMutationFamily::InstructionCapacity;
          impact.locality = HardwareMutationLocality::LocalCone;
          for (const ResizeInstructionStore &store : decision.stores) {
            addModuleRoot(impact.tech.realizationRoots, store.target);
            addModuleRoot(impact.spatial.placementRoots, store.target);
          }
        } else if constexpr (std::is_same_v<Decision,
                                            ResizeInstructionStore> ||
                             std::is_same_v<Decision, ResizeMemory> ||
                             std::is_same_v<Decision, ResizeFifo>) {
          if constexpr (std::is_same_v<Decision, ResizeInstructionStore>)
            impact.family = HardwareMutationFamily::InstructionCapacity;
          else if constexpr (std::is_same_v<Decision, ResizeMemory>)
            impact.family = HardwareMutationFamily::SpatialMemory;
          else
            impact.family = HardwareMutationFamily::SpatialFifo;
          impact.locality = HardwareMutationLocality::LocalCone;
          addModuleRoot(impact.tech.realizationRoots, decision.target);
          addModuleRoot(impact.spatial.placementRoots, decision.target);
        } else if constexpr (std::is_same_v<Decision,
                                            ChangeFifoBypassCapability>) {
          impact.family = HardwareMutationFamily::SpatialFifo;
          impact.locality = HardwareMutationLocality::LocalCone;
          impact.spatial.kind = HardwareMappingImpactKind::Reopen;
          addModuleRoot(impact.spatial.placementRoots, decision.target);
        } else if constexpr (std::is_same_v<
                                 Decision, ResizeTemporalOperandBuffer>) {
          impact.family = HardwareMutationFamily::TemporalOperandBuffer;
          impact.locality = HardwareMutationLocality::LocalCone;
          addModuleRoot(impact.spatial.placementRoots, decision.target);
        } else if constexpr (std::is_same_v<
                                 Decision,
                                 ChangeTemporalOperandBufferMode>) {
          impact.family = HardwareMutationFamily::TemporalOperandBuffer;
          impact.locality = HardwareMutationLocality::LocalCone;
          impact.spatial.kind = HardwareMappingImpactKind::Reopen;
          addModuleRoot(impact.spatial.placementRoots, decision.target);
        } else {
          if constexpr (std::is_same_v<Decision, ChangeFuInventory> ||
                        std::is_same_v<Decision, ChangeFuCapability> ||
                        std::is_same_v<Decision, ChangePeKind>)
            impact.family = HardwareMutationFamily::FuCapability;
          else if constexpr (std::is_same_v<Decision,
                                             ChangeSwitchModeOrScheduleCapacity>)
            impact.family = HardwareMutationFamily::SpatialSwitch;
          else
            impact.family = HardwareMutationFamily::SpatialMemory;
          impact.locality = HardwareMutationLocality::GlobalReopen;
          impact.tech.kind = HardwareMappingImpactKind::Reopen;
          impact.spatial.kind = HardwareMappingImpactKind::Reopen;
          addModuleRoot(impact.tech.realizationRoots, decision.target);
          addModuleRoot(impact.spatial.placementRoots, decision.target);
        }
      },
      candidate.decision);
  canonicalizeImpact(impact);
  return impact;
}

HardwareImpactProjection projectHardwareImpact(
    const SystemCompositionCandidateDecision &candidate,
    std::optional<ArtifactRootReference> child) {
  HardwareImpactProjection impact{candidate.parent, std::move(child), {}, {},
                                  {}};
  impact.family = HardwareMutationFamily::SystemAccCore;
  impact.locality = HardwareMutationLocality::GlobalReopen;
  std::visit(
      [&](const auto &decision) {
        using Decision = std::decay_t<decltype(decision)>;
        if constexpr (std::is_same_v<Decision, AddAccCore>) {
          impact.family = HardwareMutationFamily::SystemAccCore;
          impact.system.kind = HardwareMappingImpactKind::Rebase;
        } else if constexpr (std::is_same_v<Decision, RemoveAccCore> ||
                             std::is_same_v<Decision,
                                            ReplaceSpatialAttachment>) {
          impact.family = HardwareMutationFamily::SystemAccCore;
          impact.system.kind = HardwareMappingImpactKind::Reopen;
          impact.system.executionRoots.push_back(decision.target);
        } else if constexpr (std::is_same_v<
                                 Decision, SelectInstructionCoreRealization>) {
          impact.family = HardwareMutationFamily::SystemInstructionContext;
          impact.locality = HardwareMutationLocality::LocalCone;
          impact.system.kind = HardwareMappingImpactKind::Reopen;
          impact.system.instructionContextRoots.push_back(decision.target);
        } else if constexpr (std::is_same_v<Decision,
                                            ChangeTransportResource>) {
          impact.family = HardwareMutationFamily::SystemTransport;
          impact.system.kind = HardwareMappingImpactKind::Reopen;
          impact.system.transportRoots.push_back(decision.target);
        } else if constexpr (std::is_same_v<Decision,
                                            ChangeTransportConnection>) {
          impact.family = HardwareMutationFamily::SystemTransport;
          impact.locality = HardwareMutationLocality::LocalCone;
          impact.system.kind = HardwareMappingImpactKind::Reopen;
          impact.system.routeRoots.push_back(decision.destination);
          impact.system.routeRoots.push_back(decision.source);
        } else {
          impact.family = HardwareMutationFamily::SystemMemoryService;
          impact.system.kind = HardwareMappingImpactKind::Reopen;
          std::visit(
              [&](const auto &attachment) {
                using Attachment = std::decay_t<decltype(attachment)>;
                if constexpr (std::is_same_v<
                                  Attachment,
                                  ChangeSpatialMemoryAttachment>) {
                  impact.system.memoryRoots.push_back(
                      attachment.spatialEndpoint);
                  impact.system.serviceRoots.push_back(
                      attachment.serviceEndpoint);
                } else {
                  impact.system.memoryRoots.push_back(attachment.destination);
                  impact.system.memoryRoots.push_back(attachment.source);
                }
              },
              decision.value);
        }
      },
      candidate.decision);
  canonicalizeImpact(impact);
  return impact;
}

} // namespace loom::dse
