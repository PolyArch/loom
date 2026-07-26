#include "Fabric/Artifact/FabricSystemContracts.h"

#include "Fabric/IR/ResourceContractRecord.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::fabric;

namespace {

constexpr std::uint32_t kActiveHighWireTag = 0;
constexpr std::uint32_t kActiveLowWireTag = 1;
constexpr std::uint32_t kSynchronousWireTag = 0;
constexpr std::uint32_t kAsynchronousWireTag = 1;
constexpr std::uint32_t kAssertedWireTag = 0;
constexpr std::uint32_t kDeassertedWireTag = 1;
constexpr std::uint32_t kOptionalClockAbsentWireTag = 0;
constexpr std::uint32_t kOptionalClockPresentWireTag = 1;
constexpr std::uint32_t kAsyncFifoWireTag = 0;
constexpr std::uint32_t kRiscVXLenCount = 2;
constexpr std::uint32_t kRiscVBaseCount = 2;
constexpr std::uint32_t kRiscVExtensionCount = 12;
constexpr std::uint32_t kInstructionEndiannessCount = 2;
constexpr std::uint32_t kPrivilegeModeCount = 3;
constexpr std::uint32_t kRiscVAbiCount = 7;
constexpr std::uint32_t kRiscVMemoryOrderingCount = 2;
constexpr std::uint32_t kInstructionSyncScopeCount = 3;
constexpr std::uint32_t kRiscVCodeModelCount = 2;
constexpr std::uint32_t kRelocationModelCount = 2;
constexpr std::uint32_t kInstructionRuntimeServiceCount = 4;
constexpr std::uint32_t kInstructionOperationClassCount = 11;
constexpr std::uint32_t kInstructionRealizationKindCount = 2;

llvm::Error invalidContract(llvm::StringRef contract,
                            const llvm::Twine &message);
llvm::Error malformedRecord(llvm::StringRef record, const llvm::Twine &message);

class ContractRecordWriter {
public:
  void u32(std::uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
  }

  void u64(std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
  }

  llvm::Error sequenceSize(std::size_t size, llvm::StringRef field) {
    if (size > std::numeric_limits<std::uint32_t>::max())
      return invalidContract("instruction core", field + " exceeds uint32");
    u32(static_cast<std::uint32_t>(size));
    return llvm::Error::success();
  }

  void blob(llvm::ArrayRef<std::uint8_t> bytes) {
    u64(static_cast<std::uint64_t>(bytes.size()));
    bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
  }

  std::vector<std::uint8_t> take() { return std::move(bytes_); }

private:
  std::vector<std::uint8_t> bytes_;
};

class ContractRecordReader {
public:
  explicit ContractRecordReader(llvm::ArrayRef<std::uint8_t> bytes)
      : bytes_(bytes) {}

  llvm::Expected<std::uint32_t> u32(llvm::StringRef field) {
    if (bytes_.size() < 4)
      return malformedRecord(field, "truncated uint32");
    std::uint32_t value = 0;
    for (unsigned index = 0; index < 4; ++index)
      value = (value << 8) | bytes_[index];
    bytes_ = bytes_.drop_front(4);
    return value;
  }

  llvm::Expected<std::uint64_t> u64(llvm::StringRef field) {
    if (bytes_.size() < 8)
      return malformedRecord(field, "truncated uint64");
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index)
      value = (value << 8) | bytes_[index];
    bytes_ = bytes_.drop_front(8);
    return value;
  }

  llvm::Expected<std::uint32_t> tag(std::uint32_t bound,
                                    llvm::StringRef field) {
    llvm::Expected<std::uint32_t> value = u32(field);
    if (!value)
      return value.takeError();
    if (*value >= bound)
      return malformedRecord(field, llvm::Twine("unknown discriminant ") +
                                        llvm::Twine(*value));
    return *value;
  }

  llvm::Expected<std::uint32_t> count(std::size_t minimumBytesPerEntry,
                                      llvm::StringRef field) {
    llvm::Expected<std::uint32_t> value = u32(field);
    if (!value)
      return value.takeError();
    if (minimumBytesPerEntry != 0 &&
        *value > bytes_.size() / minimumBytesPerEntry)
      return malformedRecord(field, "count exceeds remaining framing");
    return *value;
  }

  llvm::Expected<llvm::ArrayRef<std::uint8_t>> blob(llvm::StringRef field) {
    llvm::Expected<std::uint64_t> size = u64(field);
    if (!size)
      return size.takeError();
    if (*size > bytes_.size())
      return malformedRecord(field, "truncated byte sequence");
    llvm::ArrayRef<std::uint8_t> result =
        bytes_.take_front(static_cast<std::size_t>(*size));
    bytes_ = bytes_.drop_front(static_cast<std::size_t>(*size));
    return result;
  }

  llvm::Error finish(llvm::StringRef record) const {
    if (!bytes_.empty())
      return malformedRecord(record, "trailing bytes");
    return llvm::Error::success();
  }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
};

bool isKnown(ResetPolarity value) {
  switch (value) {
  case ResetPolarity::ActiveHigh:
  case ResetPolarity::ActiveLow:
    return true;
  }
  return false;
}

bool isKnown(ResetTiming value) {
  switch (value) {
  case ResetTiming::Synchronous:
  case ResetTiming::Asynchronous:
    return true;
  }
  return false;
}

bool isKnown(ResetInitialState value) {
  switch (value) {
  case ResetInitialState::Asserted:
  case ResetInitialState::Deasserted:
    return true;
  }
  return false;
}

llvm::Error invalidContract(llvm::StringRef contract,
                            const llvm::Twine &message) {
  return llvm::createStringError("invalid %s contract: %s",
                                 contract.str().c_str(), message.str().c_str());
}

llvm::Error malformedRecord(llvm::StringRef record,
                            const llvm::Twine &message) {
  return makeFabricRefError(FabricRefErrorKind::MalformedSyntax,
                            llvm::Twine("invalid ") + record +
                                " record: " + message);
}

std::uint32_t resetPolarityWireTag(ResetPolarity value) {
  switch (value) {
  case ResetPolarity::ActiveHigh:
    return kActiveHighWireTag;
  case ResetPolarity::ActiveLow:
    return kActiveLowWireTag;
  }
  llvm_unreachable("validated reset polarity has no wire tag");
}

llvm::Expected<ResetPolarity> readResetPolarity(FabricByteReader &reader) {
  llvm::Expected<std::uint32_t> raw = reader.tag();
  if (!raw)
    return raw.takeError();
  switch (*raw) {
  case kActiveHighWireTag:
    return ResetPolarity::ActiveHigh;
  case kActiveLowWireTag:
    return ResetPolarity::ActiveLow;
  default:
    return malformedRecord("reset polarity",
                           llvm::Twine("unknown discriminant ") +
                               llvm::Twine(*raw));
  }
}

std::uint32_t resetTimingWireTag(ResetTiming value) {
  switch (value) {
  case ResetTiming::Synchronous:
    return kSynchronousWireTag;
  case ResetTiming::Asynchronous:
    return kAsynchronousWireTag;
  }
  llvm_unreachable("validated reset timing has no wire tag");
}

llvm::Expected<ResetTiming> readResetTiming(FabricByteReader &reader,
                                            llvm::StringRef field) {
  llvm::Expected<std::uint32_t> raw = reader.tag();
  if (!raw)
    return raw.takeError();
  switch (*raw) {
  case kSynchronousWireTag:
    return ResetTiming::Synchronous;
  case kAsynchronousWireTag:
    return ResetTiming::Asynchronous;
  default:
    return malformedRecord(field, llvm::Twine("unknown discriminant ") +
                                      llvm::Twine(*raw));
  }
}

std::uint32_t resetInitialStateWireTag(ResetInitialState value) {
  switch (value) {
  case ResetInitialState::Asserted:
    return kAssertedWireTag;
  case ResetInitialState::Deasserted:
    return kDeassertedWireTag;
  }
  llvm_unreachable("validated reset initial state has no wire tag");
}

llvm::Expected<ResetInitialState>
readResetInitialState(FabricByteReader &reader) {
  llvm::Expected<std::uint32_t> raw = reader.tag();
  if (!raw)
    return raw.takeError();
  switch (*raw) {
  case kAssertedWireTag:
    return ResetInitialState::Asserted;
  case kDeassertedWireTag:
    return ResetInitialState::Deasserted;
  default:
    return malformedRecord("reset initial state",
                           llvm::Twine("unknown discriminant ") +
                               llvm::Twine(*raw));
  }
}

llvm::Error readAsyncFifoVariant(FabricByteReader &reader) {
  llvm::Expected<std::uint32_t> raw = reader.tag();
  if (!raw)
    return raw.takeError();
  if (*raw == kAsyncFifoWireTag)
    return llvm::Error::success();
  return malformedRecord("clock crossing variant",
                         llvm::Twine("unknown discriminant ") +
                             llvm::Twine(*raw));
}

llvm::Error requireFinished(FabricByteReader &reader, llvm::StringRef record) {
  if (!reader.empty())
    return malformedRecord(record, "trailing bytes");
  return llvm::Error::success();
}

template <typename Record, typename Encoder>
llvm::Error requireCanonical(llvm::ArrayRef<std::uint8_t> bytes,
                             const Record &record, Encoder encode,
                             llvm::StringRef name) {
  llvm::Expected<std::vector<std::uint8_t>> canonical = encode(record);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*canonical) != bytes)
    return malformedRecord(name, "noncanonical encoding");
  return llvm::Error::success();
}

template <typename Enum>
bool isKnownClosedValue(Enum value, std::uint32_t bound) {
  return static_cast<std::uint32_t>(value) < bound;
}

template <typename Enum>
llvm::Error normalizeClosedSet(std::vector<Enum> &values, std::uint32_t bound,
                               llvm::StringRef field, bool allowEmpty) {
  for (Enum value : values)
    if (!isKnownClosedValue(value, bound))
      return invalidContract("instruction core",
                             field + " contains an unknown value");
  std::sort(values.begin(), values.end(), [](Enum lhs, Enum rhs) {
    return static_cast<std::uint32_t>(lhs) < static_cast<std::uint32_t>(rhs);
  });
  if (!allowEmpty && values.empty())
    return invalidContract("instruction core", field + " must not be empty");
  if (std::adjacent_find(values.begin(), values.end()) != values.end())
    return invalidContract("instruction core",
                           field + " contains a duplicate value");
  return llvm::Error::success();
}

template <typename Enum>
bool contains(llvm::ArrayRef<Enum> values, Enum value) {
  return std::binary_search(values.begin(), values.end(), value,
                            [](Enum lhs, Enum rhs) {
                              return static_cast<std::uint32_t>(lhs) <
                                     static_cast<std::uint32_t>(rhs);
                            });
}

template <typename Enum>
llvm::Error writeClosedSet(ContractRecordWriter &writer,
                           llvm::ArrayRef<Enum> values, llvm::StringRef field) {
  if (llvm::Error error = writer.sequenceSize(values.size(), field))
    return error;
  for (Enum value : values)
    writer.u32(static_cast<std::uint32_t>(value));
  return llvm::Error::success();
}

template <typename Enum>
llvm::Expected<std::vector<Enum>> readClosedSet(ContractRecordReader &reader,
                                                std::uint32_t bound,
                                                llvm::StringRef field) {
  llvm::Expected<std::uint32_t> size = reader.count(4, field);
  if (!size)
    return size.takeError();
  std::vector<Enum> values;
  values.reserve(*size);
  for (std::uint32_t index = 0; index < *size; ++index) {
    llvm::Expected<std::uint32_t> value = reader.tag(bound, field);
    if (!value)
      return value.takeError();
    values.push_back(static_cast<Enum>(*value));
  }
  return values;
}

bool isFloatingAbi(RiscVAbi abi) {
  return abi == RiscVAbi::Ilp32f || abi == RiscVAbi::Lp64f;
}

bool isDoubleAbi(RiscVAbi abi) {
  return abi == RiscVAbi::Ilp32d || abi == RiscVAbi::Lp64d;
}

bool is64BitAbi(RiscVAbi abi) {
  return abi == RiscVAbi::Lp64 || abi == RiscVAbi::Lp64f ||
         abi == RiscVAbi::Lp64d;
}

llvm::Expected<std::vector<ExecutionUnitRecord>>
normalizeExecutionUnits(std::vector<ExecutionUnitRecord> units) {
  if (units.empty())
    return invalidContract("instruction core microarchitecture",
                           "execution_units must not be empty");
  for (const ExecutionUnitRecord &unit : units) {
    if (!isKnownClosedValue(unit.operationClass,
                            kInstructionOperationClassCount))
      return invalidContract("instruction core microarchitecture",
                             "execution unit has an unknown operation class");
    if (unit.count == 0 || unit.latencyCycles == 0 ||
        unit.initiationInterval == 0)
      return invalidContract(
          "instruction core microarchitecture",
          "execution-unit count, latency, and initiation interval must be "
          "positive");
  }

  auto key = [](const ExecutionUnitRecord &unit) {
    return std::make_tuple(static_cast<std::uint32_t>(unit.operationClass),
                           unit.latencyCycles, unit.initiationInterval);
  };
  std::sort(units.begin(), units.end(), [&](const auto &lhs, const auto &rhs) {
    return key(lhs) < key(rhs);
  });

  std::vector<ExecutionUnitRecord> normalized;
  normalized.reserve(units.size());
  for (const ExecutionUnitRecord &unit : units) {
    if (normalized.empty() || key(normalized.back()) != key(unit)) {
      normalized.push_back(unit);
      continue;
    }
    if (normalized.back().count >
        std::numeric_limits<std::uint32_t>::max() - unit.count)
      return invalidContract("instruction core microarchitecture",
                             "execution-unit count overflows uint32");
    normalized.back().count += unit.count;
  }
  return normalized;
}

llvm::Error validateInstructionResourceContract(
    const ::fabric::ResourceContract &contract) {
  if (contract.stateCount() == 0 || contract.requesterCount() != 1 ||
      contract.usePatternCount() == 0)
    return invalidContract(
        "instruction core microarchitecture",
        "resource contract requires state, one requester, and a use pattern");
  for (std::uint32_t state = 0; state < contract.stateCount(); ++state)
    for (const ::fabric::CapacityDimension &dimension :
         contract.capacityDimensions(::fabric::StateKey(state)))
      if (dimension.initialOccupancy != ::fabric::CapacityUnits(0))
        return invalidContract("instruction core microarchitecture",
                               "resource contract initial state is not free");
  return llvm::Error::success();
}

template <typename Pipeline> bool allPositive(const Pipeline &pipeline);

template <>
bool allPositive(const InOrderMicroarchitectureDeclaration &pipeline) {
  return pipeline.fetchWidth != 0 && pipeline.decodeWidth != 0 &&
         pipeline.issueWidth != 0 && pipeline.commitWidth != 0 &&
         pipeline.memoryIssueWidth != 0 && pipeline.memoryCommitWidth != 0 &&
         pipeline.maxOutstandingMemoryOperations != 0 &&
         pipeline.storeBufferEntries != 0;
}

template <>
bool allPositive(const OutOfOrderMicroarchitectureDeclaration &pipeline) {
  return pipeline.fetchWidth != 0 && pipeline.decodeWidth != 0 &&
         pipeline.renameWidth != 0 && pipeline.dispatchWidth != 0 &&
         pipeline.issueWidth != 0 && pipeline.writebackWidth != 0 &&
         pipeline.commitWidth != 0 && pipeline.reorderBufferEntries != 0 &&
         pipeline.issueQueueEntries != 0 && pipeline.loadQueueEntries != 0 &&
         pipeline.storeQueueEntries != 0 &&
         pipeline.physicalIntegerRegisters != 0 &&
         pipeline.physicalFloatRegisters != 0 &&
         pipeline.physicalVectorRegisters != 0;
}

} // namespace

std::vector<std::uint8_t> loom::fabric::encodeFabricImportedModuleTargetRef(
    const FabricImportedModuleTargetRef &reference) {
  FabricByteWriter writer;
  writer.field(reference.dependencyOrdinal);
  encodeFabricRef(writer, reference.target);
  return writer.take();
}

llvm::Expected<FabricImportedModuleTargetRef>
loom::fabric::decodeFabricImportedModuleTargetRef(
    llvm::ArrayRef<std::uint8_t> bytes) {
  FabricByteReader reader(bytes);
  llvm::Expected<std::uint64_t> dependencyOrdinal = reader.field();
  if (!dependencyOrdinal)
    return dependencyOrdinal.takeError();
  FabricModuleTemplateRef target;
  if (llvm::Error error = decodeFabricRefInto(reader, target))
    return std::move(error);
  if (llvm::Error error = requireFinished(reader, "ImportedModule target"))
    return std::move(error);
  return FabricImportedModuleTargetRef{*dependencyOrdinal, target};
}

std::vector<std::uint8_t>
loom::fabric::encodeFabricImportedModuleBoundaryEndpointRef(
    const FabricImportedModuleBoundaryEndpointRef &reference) {
  FabricByteWriter writer;
  writer.field(reference.dependencyOrdinal);
  encodeFabricRef(writer, reference.target);
  return writer.take();
}

llvm::Expected<FabricImportedModuleBoundaryEndpointRef>
loom::fabric::decodeFabricImportedModuleBoundaryEndpointRef(
    llvm::ArrayRef<std::uint8_t> bytes) {
  FabricByteReader reader(bytes);
  llvm::Expected<std::uint64_t> dependencyOrdinal = reader.field();
  if (!dependencyOrdinal)
    return dependencyOrdinal.takeError();
  FabricModuleBoundaryEndpointRef target;
  if (llvm::Error error = decodeFabricRefInto(reader, target))
    return std::move(error);
  if (llvm::Error error =
          requireFinished(reader, "ImportedModule boundary endpoint"))
    return std::move(error);
  return FabricImportedModuleBoundaryEndpointRef{*dependencyOrdinal, target};
}

llvm::Expected<FabricSpatialAttachmentEndpointRef>
FabricSpatialAttachmentEndpointRef::create(
    FabricTransportEndpointRef endpoint) {
  if (endpoint.owner.kind() !=
      FabricTransportEndpointOwnerKind::SpatialCoreOccurrence)
    return makeFabricRefError(
        FabricRefErrorKind::WrongOwner,
        "a Spatial attachment transport endpoint must be owned by the "
        "AccCore's SpatialCore occurrence");
  return FabricSpatialAttachmentEndpointRef(Endpoint(std::move(endpoint)));
}

llvm::Expected<FabricSpatialAttachmentEndpointRef>
FabricSpatialAttachmentEndpointRef::create(FabricMemoryEndpointRef endpoint) {
  if (endpoint.owner.kind() !=
      FabricMemoryEndpointOwnerKind::SpatialCoreOccurrence)
    return makeFabricRefError(
        FabricRefErrorKind::WrongOwner,
        "a Spatial attachment memory endpoint must be owned by the AccCore's "
        "SpatialCore occurrence");
  return FabricSpatialAttachmentEndpointRef(Endpoint(std::move(endpoint)));
}

std::vector<std::uint8_t>
loom::fabric::encodeFabricSpatialAttachmentEndpointRef(
    const FabricSpatialAttachmentEndpointRef &reference) {
  FabricByteWriter writer;
  writer.tag(static_cast<std::uint32_t>(reference.plane()));
  if (const FabricTransportEndpointRef *endpoint = reference.transport())
    encodeFabricRef(writer, *endpoint);
  else
    encodeFabricRef(writer, *reference.memory());
  return writer.take();
}

llvm::Expected<FabricSpatialAttachmentEndpointRef>
loom::fabric::decodeFabricSpatialAttachmentEndpointRef(
    llvm::ArrayRef<std::uint8_t> bytes) {
  FabricByteReader reader(bytes);
  llvm::Expected<std::uint32_t> plane =
      readFabricClosedTag(reader, 2, "spatial attachment endpoint plane");
  if (!plane)
    return plane.takeError();

  llvm::Expected<FabricSpatialAttachmentEndpointRef> endpoint =
      [&]() -> llvm::Expected<FabricSpatialAttachmentEndpointRef> {
    if (*plane == static_cast<std::uint32_t>(
                      FabricSpatialAttachmentEndpointRef::Plane::Transport)) {
      FabricTransportEndpointRef transport;
      if (llvm::Error error = decodeFabricRefInto(reader, transport))
        return std::move(error);
      return FabricSpatialAttachmentEndpointRef::create(std::move(transport));
    }
    FabricMemoryEndpointRef memory;
    if (llvm::Error error = decodeFabricRefInto(reader, memory))
      return std::move(error);
    return FabricSpatialAttachmentEndpointRef::create(std::move(memory));
  }();
  if (!endpoint)
    return endpoint.takeError();
  if (llvm::Error error =
          requireFinished(reader, "Spatial attachment endpoint"))
    return std::move(error);
  return std::move(*endpoint);
}

llvm::Expected<SystemTransferPatternRecord> SystemTransferPatternRecord::create(
    FabricTransferPatternRef pattern, FabricTransportEndpointRef ingress,
    std::vector<FabricTransportEndpointRef> egresses,
    FabricUsePatternRef usePattern) {
  const SystemTransportResourceRef resource = pattern.resource;
  const auto *ingressOwner =
      std::get_if<SystemTransportResourceRef>(&ingress.owner.payload);
  if (!ingressOwner || *ingressOwner != resource)
    return invalidContract(
        "system transfer pattern",
        "ingress is not owned by the pattern's transport resource");
  if (egresses.empty())
    return invalidContract("system transfer pattern",
                           "egress set must not be empty");

  using EncodedEndpoint =
      std::pair<std::vector<std::uint8_t>, FabricTransportEndpointRef>;
  std::vector<EncodedEndpoint> canonicalEgresses;
  canonicalEgresses.reserve(egresses.size());
  for (FabricTransportEndpointRef &egress : egresses) {
    const auto *egressOwner =
        std::get_if<SystemTransportResourceRef>(&egress.owner.payload);
    if (!egressOwner || *egressOwner != resource)
      return invalidContract(
          "system transfer pattern",
          "egress is not owned by the pattern's transport resource");
    canonicalEgresses.emplace_back(canonicalFabricBytes(egress),
                                   std::move(egress));
  }
  llvm::sort(canonicalEgresses,
             [](const EncodedEndpoint &lhs, const EncodedEndpoint &rhs) {
               return lhs.first < rhs.first;
             });
  for (std::size_t index = 1; index < canonicalEgresses.size(); ++index)
    if (canonicalEgresses[index - 1].first == canonicalEgresses[index].first)
      return invalidContract("system transfer pattern",
                             "egress set contains a duplicate endpoint");

  const FabricInventoryOwnerRef &useOwner = usePattern.owner.catalog();
  const auto *useResource =
      std::get_if<SystemTransportResourceRef>(&useOwner.payload);
  if (!useResource || *useResource != resource)
    return invalidContract(
        "system transfer pattern",
        "UsePattern is not owned by the pattern's transport resource");

  egresses.clear();
  egresses.reserve(canonicalEgresses.size());
  for (EncodedEndpoint &entry : canonicalEgresses)
    egresses.push_back(std::move(entry.second));
  return SystemTransferPatternRecord(std::move(pattern), std::move(ingress),
                                     std::move(egresses),
                                     std::move(usePattern));
}

std::vector<std::uint8_t> loom::fabric::encodeSystemTransferPatternRecord(
    const SystemTransferPatternRecord &record) {
  FabricByteWriter writer;
  encodeFabricRef(writer, record.pattern());
  encodeFabricRef(writer, record.ingress());
  writer.field(record.egresses().size());
  for (const FabricTransportEndpointRef &egress : record.egresses())
    encodeFabricRef(writer, egress);
  encodeFabricRef(writer, record.usePattern());
  return writer.take();
}

llvm::Expected<SystemTransferPatternRecord>
loom::fabric::decodeSystemTransferPatternRecord(
    llvm::ArrayRef<std::uint8_t> bytes) {
  FabricByteReader reader(bytes);
  FabricTransferPatternRef pattern;
  if (llvm::Error error = decodeFabricRefInto(reader, pattern))
    return std::move(error);
  FabricTransportEndpointRef ingress;
  if (llvm::Error error = decodeFabricRefInto(reader, ingress))
    return std::move(error);
  llvm::Expected<std::uint64_t> count = reader.field();
  if (!count)
    return count.takeError();
  if (*count > bytes.size() / 8)
    return malformedRecord("system transfer pattern",
                           "egress count exceeds remaining framing");
  std::vector<FabricTransportEndpointRef> egresses;
  egresses.reserve(static_cast<std::size_t>(*count));
  for (std::uint64_t index = 0; index < *count; ++index) {
    FabricTransportEndpointRef egress;
    if (llvm::Error error = decodeFabricRefInto(reader, egress))
      return std::move(error);
    egresses.push_back(std::move(egress));
  }
  FabricUsePatternRef usePattern;
  if (llvm::Error error = decodeFabricRefInto(reader, usePattern))
    return std::move(error);
  if (llvm::Error error = requireFinished(reader, "system transfer pattern"))
    return std::move(error);

  auto record = SystemTransferPatternRecord::create(
      std::move(pattern), std::move(ingress), std::move(egresses),
      std::move(usePattern));
  if (!record)
    return record.takeError();
  const std::vector<std::uint8_t> canonical =
      encodeSystemTransferPatternRecord(*record);
  if (llvm::ArrayRef<std::uint8_t>(canonical) != bytes)
    return malformedRecord("system transfer pattern",
                           "record is not in canonical order");
  return std::move(*record);
}

llvm::Expected<InstructionCoreArchitecturalContract>
InstructionCoreArchitecturalContract::create(
    RiscVArchitectureDeclaration declaration) {
  if (!isKnownClosedValue(declaration.xlen, kRiscVXLenCount) ||
      !isKnownClosedValue(declaration.base, kRiscVBaseCount) ||
      !isKnownClosedValue(declaration.endianness,
                          kInstructionEndiannessCount) ||
      !isKnownClosedValue(declaration.memoryOrdering,
                          kRiscVMemoryOrderingCount))
    return invalidContract("instruction core architecture",
                           "unknown closed-enum value");

  if (llvm::Error error = normalizeClosedSet(declaration.extensions,
                                             kRiscVExtensionCount, "extensions",
                                             /*allowEmpty=*/true))
    return std::move(error);
  if (llvm::Error error =
          normalizeClosedSet(declaration.privilegeModes, kPrivilegeModeCount,
                             "privilege_modes", /*allowEmpty=*/false))
    return std::move(error);
  if (llvm::Error error =
          normalizeClosedSet(declaration.abiCapabilities, kRiscVAbiCount,
                             "abi_capabilities", /*allowEmpty=*/false))
    return std::move(error);
  if (llvm::Error error = normalizeClosedSet(
          declaration.syncScopes, kInstructionSyncScopeCount, "sync_scopes",
          /*allowEmpty=*/false))
    return std::move(error);
  if (llvm::Error error = normalizeClosedSet(
          declaration.codeModels, kRiscVCodeModelCount, "code_models",
          /*allowEmpty=*/false))
    return std::move(error);
  if (llvm::Error error =
          normalizeClosedSet(declaration.relocationModels,
                             kRelocationModelCount, "relocation_models",
                             /*allowEmpty=*/false))
    return std::move(error);
  if (llvm::Error error = normalizeClosedSet(
          declaration.runtimeServices, kInstructionRuntimeServiceCount,
          "runtime_services", /*allowEmpty=*/true))
    return std::move(error);

  const std::uint32_t xlen = declaration.xlen == RiscVXLen::X32 ? 32u : 64u;
  if (declaration.physicalAddressWidthBits == 0 ||
      declaration.physicalAddressWidthBits > xlen)
    return invalidContract(
        "instruction core architecture",
        "physical_address_width_bits must be positive and no greater than "
        "xlen");
  if (declaration.base == RiscVBase::E && declaration.xlen != RiscVXLen::X32)
    return invalidContract("instruction core architecture",
                           "base E requires X32");
  if (!contains<PrivilegeMode>(declaration.privilegeModes,
                               PrivilegeMode::Machine))
    return invalidContract("instruction core architecture",
                           "privilege_modes must include Machine");
  if (contains<RiscVExtension>(declaration.extensions, RiscVExtension::D) &&
      !contains<RiscVExtension>(declaration.extensions, RiscVExtension::F))
    return invalidContract("instruction core architecture",
                           "extension D requires extension F");
  if (declaration.memoryOrdering == RiscVMemoryOrdering::Ztso &&
      !contains<RiscVExtension>(declaration.extensions, RiscVExtension::Ztso))
    return invalidContract("instruction core architecture",
                           "Ztso ordering requires extension Ztso");

  for (RiscVAbi abi : declaration.abiCapabilities) {
    if (is64BitAbi(abi) != (declaration.xlen == RiscVXLen::X64))
      return invalidContract("instruction core architecture",
                             "ABI width does not match xlen");
    if (abi == RiscVAbi::Ilp32e) {
      if (declaration.base != RiscVBase::E)
        return invalidContract("instruction core architecture",
                               "Ilp32e requires base E");
    } else if (declaration.base != RiscVBase::I) {
      return invalidContract("instruction core architecture",
                             "non-Ilp32e ABI requires base I");
    }
    if (isFloatingAbi(abi) &&
        !contains<RiscVExtension>(declaration.extensions, RiscVExtension::F))
      return invalidContract("instruction core architecture",
                             "floating ABI requires extension F");
    if (isDoubleAbi(abi) &&
        !contains<RiscVExtension>(declaration.extensions, RiscVExtension::D))
      return invalidContract("instruction core architecture",
                             "double ABI requires extension D");
  }

  return InstructionCoreArchitecturalContract(std::move(declaration));
}

llvm::Expected<InstructionCoreMicroarchitecturalRealization>
InstructionCoreMicroarchitecturalRealization::createInOrder(
    InstructionCoreCommonDeclaration common,
    InOrderMicroarchitectureDeclaration pipeline) {
  if (common.hardwareThreadCount == 0 || !allPositive(pipeline))
    return invalidContract("instruction core microarchitecture",
                           "thread count, widths, and capacities must be "
                           "positive");
  llvm::Expected<std::vector<ExecutionUnitRecord>> units =
      normalizeExecutionUnits(std::move(common.executionUnits));
  if (!units)
    return units.takeError();
  if (llvm::Error error =
          validateInstructionResourceContract(common.resourceContract))
    return std::move(error);
  return InstructionCoreMicroarchitecturalRealization(
      InstructionCoreRealizationKind::InOrder, common.hardwareThreadCount,
      std::move(*units), std::move(common.resourceContract), pipeline);
}

llvm::Expected<InstructionCoreMicroarchitecturalRealization>
InstructionCoreMicroarchitecturalRealization::createOutOfOrder(
    InstructionCoreCommonDeclaration common,
    OutOfOrderMicroarchitectureDeclaration pipeline) {
  if (common.hardwareThreadCount == 0 || !allPositive(pipeline))
    return invalidContract("instruction core microarchitecture",
                           "thread count, widths, and capacities must be "
                           "positive");
  llvm::Expected<std::vector<ExecutionUnitRecord>> units =
      normalizeExecutionUnits(std::move(common.executionUnits));
  if (!units)
    return units.takeError();
  if (llvm::Error error =
          validateInstructionResourceContract(common.resourceContract))
    return std::move(error);
  return InstructionCoreMicroarchitecturalRealization(
      InstructionCoreRealizationKind::OutOfOrder, common.hardwareThreadCount,
      std::move(*units), std::move(common.resourceContract), pipeline);
}

llvm::Expected<std::vector<std::uint8_t>>
loom::fabric::encodeInstructionCoreArchitecturalContract(
    const InstructionCoreArchitecturalContract &contract) {
  ContractRecordWriter writer;
  writer.u32(0); // RiscV architecture variant.
  writer.u32(static_cast<std::uint32_t>(contract.xlen()));
  writer.u32(static_cast<std::uint32_t>(contract.base()));
  if (llvm::Error error =
          writeClosedSet(writer, contract.extensions(), "extensions"))
    return std::move(error);
  writer.u32(static_cast<std::uint32_t>(contract.endianness()));
  writer.u32(contract.physicalAddressWidthBits());
  if (llvm::Error error =
          writeClosedSet(writer, contract.privilegeModes(), "privilege_modes"))
    return std::move(error);
  if (llvm::Error error = writeClosedSet(writer, contract.abiCapabilities(),
                                         "abi_capabilities"))
    return std::move(error);
  writer.u32(static_cast<std::uint32_t>(contract.memoryOrdering()));
  if (llvm::Error error =
          writeClosedSet(writer, contract.syncScopes(), "sync_scopes"))
    return std::move(error);
  if (llvm::Error error =
          writeClosedSet(writer, contract.codeModels(), "code_models"))
    return std::move(error);
  if (llvm::Error error = writeClosedSet(writer, contract.relocationModels(),
                                         "relocation_models"))
    return std::move(error);
  if (llvm::Error error = writeClosedSet(writer, contract.runtimeServices(),
                                         "runtime_services"))
    return std::move(error);
  return writer.take();
}

llvm::Expected<InstructionCoreArchitecturalContract>
loom::fabric::decodeInstructionCoreArchitecturalContract(
    llvm::ArrayRef<std::uint8_t> bytes) {
  ContractRecordReader reader(bytes);
  llvm::Expected<std::uint32_t> variant =
      reader.tag(1, "instruction architecture variant");
  if (!variant)
    return variant.takeError();

  RiscVArchitectureDeclaration declaration;
  llvm::Expected<std::uint32_t> xlen = reader.tag(kRiscVXLenCount, "xlen");
  if (!xlen)
    return xlen.takeError();
  declaration.xlen = static_cast<RiscVXLen>(*xlen);
  llvm::Expected<std::uint32_t> base = reader.tag(kRiscVBaseCount, "base");
  if (!base)
    return base.takeError();
  declaration.base = static_cast<RiscVBase>(*base);
  llvm::Expected<std::vector<RiscVExtension>> extensions =
      readClosedSet<RiscVExtension>(reader, kRiscVExtensionCount, "extensions");
  if (!extensions)
    return extensions.takeError();
  declaration.extensions = std::move(*extensions);
  llvm::Expected<std::uint32_t> endianness =
      reader.tag(kInstructionEndiannessCount, "endianness");
  if (!endianness)
    return endianness.takeError();
  declaration.endianness = static_cast<InstructionEndianness>(*endianness);
  llvm::Expected<std::uint32_t> addressWidth =
      reader.u32("physical address width");
  if (!addressWidth)
    return addressWidth.takeError();
  declaration.physicalAddressWidthBits = *addressWidth;

  llvm::Expected<std::vector<PrivilegeMode>> privileges =
      readClosedSet<PrivilegeMode>(reader, kPrivilegeModeCount,
                                   "privilege modes");
  if (!privileges)
    return privileges.takeError();
  declaration.privilegeModes = std::move(*privileges);
  llvm::Expected<std::vector<RiscVAbi>> abis =
      readClosedSet<RiscVAbi>(reader, kRiscVAbiCount, "ABI capabilities");
  if (!abis)
    return abis.takeError();
  declaration.abiCapabilities = std::move(*abis);
  llvm::Expected<std::uint32_t> ordering =
      reader.tag(kRiscVMemoryOrderingCount, "memory ordering");
  if (!ordering)
    return ordering.takeError();
  declaration.memoryOrdering = static_cast<RiscVMemoryOrdering>(*ordering);
  llvm::Expected<std::vector<InstructionSyncScope>> scopes =
      readClosedSet<InstructionSyncScope>(reader, kInstructionSyncScopeCount,
                                          "sync scopes");
  if (!scopes)
    return scopes.takeError();
  declaration.syncScopes = std::move(*scopes);
  llvm::Expected<std::vector<RiscVCodeModel>> codeModels =
      readClosedSet<RiscVCodeModel>(reader, kRiscVCodeModelCount,
                                    "code models");
  if (!codeModels)
    return codeModels.takeError();
  declaration.codeModels = std::move(*codeModels);
  llvm::Expected<std::vector<RelocationModel>> relocations =
      readClosedSet<RelocationModel>(reader, kRelocationModelCount,
                                     "relocation models");
  if (!relocations)
    return relocations.takeError();
  declaration.relocationModels = std::move(*relocations);
  llvm::Expected<std::vector<InstructionRuntimeService>> services =
      readClosedSet<InstructionRuntimeService>(
          reader, kInstructionRuntimeServiceCount, "runtime services");
  if (!services)
    return services.takeError();
  declaration.runtimeServices = std::move(*services);
  if (llvm::Error error = reader.finish("instruction architecture"))
    return std::move(error);

  llvm::Expected<InstructionCoreArchitecturalContract> contract =
      InstructionCoreArchitecturalContract::create(std::move(declaration));
  if (!contract)
    return contract.takeError();
  if (llvm::Error error = requireCanonical(
          bytes, *contract, encodeInstructionCoreArchitecturalContract,
          "instruction architecture"))
    return std::move(error);
  return std::move(*contract);
}

llvm::Expected<std::vector<std::uint8_t>>
loom::fabric::encodeInstructionCoreMicroarchitecturalRealization(
    const InstructionCoreMicroarchitecturalRealization &realization) {
  ContractRecordWriter writer;
  writer.u32(static_cast<std::uint32_t>(realization.kind()));
  writer.u32(realization.hardwareThreadCount());
  if (llvm::Error error = writer.sequenceSize(
          realization.executionUnits().size(), "execution units"))
    return std::move(error);
  for (const ExecutionUnitRecord &unit : realization.executionUnits()) {
    writer.u32(static_cast<std::uint32_t>(unit.operationClass));
    writer.u32(unit.count);
    writer.u32(unit.latencyCycles);
    writer.u32(unit.initiationInterval);
  }
  llvm::Expected<std::vector<std::uint8_t>> resource =
      ::fabric::encodeResourceContractRecord(realization.resourceContract());
  if (!resource)
    return resource.takeError();
  writer.blob(*resource);

  if (const auto *pipeline = realization.inOrder()) {
    writer.u32(pipeline->fetchWidth);
    writer.u32(pipeline->decodeWidth);
    writer.u32(pipeline->issueWidth);
    writer.u32(pipeline->commitWidth);
    writer.u32(pipeline->memoryIssueWidth);
    writer.u32(pipeline->memoryCommitWidth);
    writer.u32(pipeline->maxOutstandingMemoryOperations);
    writer.u32(pipeline->storeBufferEntries);
  } else {
    const auto &outOfOrder = *realization.outOfOrder();
    writer.u32(outOfOrder.fetchWidth);
    writer.u32(outOfOrder.decodeWidth);
    writer.u32(outOfOrder.renameWidth);
    writer.u32(outOfOrder.dispatchWidth);
    writer.u32(outOfOrder.issueWidth);
    writer.u32(outOfOrder.writebackWidth);
    writer.u32(outOfOrder.commitWidth);
    writer.u32(outOfOrder.reorderBufferEntries);
    writer.u32(outOfOrder.issueQueueEntries);
    writer.u32(outOfOrder.loadQueueEntries);
    writer.u32(outOfOrder.storeQueueEntries);
    writer.u32(outOfOrder.physicalIntegerRegisters);
    writer.u32(outOfOrder.physicalFloatRegisters);
    writer.u32(outOfOrder.physicalVectorRegisters);
  }
  return writer.take();
}

llvm::Expected<InstructionCoreMicroarchitecturalRealization>
loom::fabric::decodeInstructionCoreMicroarchitecturalRealization(
    llvm::ArrayRef<std::uint8_t> bytes) {
  ContractRecordReader reader(bytes);
  llvm::Expected<std::uint32_t> kind =
      reader.tag(kInstructionRealizationKindCount,
                 "instruction microarchitecture variant");
  if (!kind)
    return kind.takeError();
  llvm::Expected<std::uint32_t> threads = reader.u32("hardware thread count");
  if (!threads)
    return threads.takeError();
  llvm::Expected<std::uint32_t> unitCount =
      reader.count(16, "execution unit count");
  if (!unitCount)
    return unitCount.takeError();
  std::vector<ExecutionUnitRecord> units;
  units.reserve(*unitCount);
  for (std::uint32_t index = 0; index < *unitCount; ++index) {
    llvm::Expected<std::uint32_t> operation = reader.tag(
        kInstructionOperationClassCount, "execution-unit operation class");
    if (!operation)
      return operation.takeError();
    llvm::Expected<std::uint32_t> count = reader.u32("execution-unit count");
    if (!count)
      return count.takeError();
    llvm::Expected<std::uint32_t> latency =
        reader.u32("execution-unit latency");
    if (!latency)
      return latency.takeError();
    llvm::Expected<std::uint32_t> interval =
        reader.u32("execution-unit initiation interval");
    if (!interval)
      return interval.takeError();
    units.push_back(
        ExecutionUnitRecord{static_cast<InstructionOperationClass>(*operation),
                            *count, *latency, *interval});
  }
  llvm::Expected<llvm::ArrayRef<std::uint8_t>> resourceBytes =
      reader.blob("resource contract");
  if (!resourceBytes)
    return resourceBytes.takeError();
  llvm::Expected<::fabric::ResourceContract> resource =
      ::fabric::decodeResourceContractRecord(*resourceBytes);
  if (!resource)
    return resource.takeError();

  InstructionCoreCommonDeclaration common{*threads, std::move(units),
                                          std::move(*resource)};
  llvm::Expected<InstructionCoreMicroarchitecturalRealization> realization =
      [&]() -> llvm::Expected<InstructionCoreMicroarchitecturalRealization> {
    if (*kind ==
        static_cast<std::uint32_t>(InstructionCoreRealizationKind::InOrder)) {
      InOrderMicroarchitectureDeclaration pipeline;
      std::uint32_t *fields[] = {
          &pipeline.fetchWidth,
          &pipeline.decodeWidth,
          &pipeline.issueWidth,
          &pipeline.commitWidth,
          &pipeline.memoryIssueWidth,
          &pipeline.memoryCommitWidth,
          &pipeline.maxOutstandingMemoryOperations,
          &pipeline.storeBufferEntries,
      };
      for (std::uint32_t *field : fields) {
        llvm::Expected<std::uint32_t> value = reader.u32("in-order field");
        if (!value)
          return value.takeError();
        *field = *value;
      }
      return InstructionCoreMicroarchitecturalRealization::createInOrder(
          std::move(common), pipeline);
    }

    OutOfOrderMicroarchitectureDeclaration pipeline;
    std::uint32_t *fields[] = {
        &pipeline.fetchWidth,
        &pipeline.decodeWidth,
        &pipeline.renameWidth,
        &pipeline.dispatchWidth,
        &pipeline.issueWidth,
        &pipeline.writebackWidth,
        &pipeline.commitWidth,
        &pipeline.reorderBufferEntries,
        &pipeline.issueQueueEntries,
        &pipeline.loadQueueEntries,
        &pipeline.storeQueueEntries,
        &pipeline.physicalIntegerRegisters,
        &pipeline.physicalFloatRegisters,
        &pipeline.physicalVectorRegisters,
    };
    for (std::uint32_t *field : fields) {
      llvm::Expected<std::uint32_t> value = reader.u32("out-of-order field");
      if (!value)
        return value.takeError();
      *field = *value;
    }
    return InstructionCoreMicroarchitecturalRealization::createOutOfOrder(
        std::move(common), pipeline);
  }();
  if (!realization)
    return realization.takeError();
  if (llvm::Error error = reader.finish("instruction microarchitecture"))
    return std::move(error);
  if (llvm::Error error =
          requireCanonical(bytes, *realization,
                           encodeInstructionCoreMicroarchitecturalRealization,
                           "instruction microarchitecture"))
    return std::move(error);
  return std::move(*realization);
}

llvm::Expected<ClockDomainContractRecord>
ClockDomainContractRecord::create(std::uint64_t periodFs,
                                  std::uint64_t phaseFs) {
  if (periodFs == 0)
    return invalidContract("clock", "period_fs must be positive");
  if (phaseFs >= periodFs)
    return invalidContract("clock", "phase_fs must be less than period_fs");
  return ClockDomainContractRecord(periodFs, phaseFs);
}

llvm::Expected<ResetDomainContractRecord> ResetDomainContractRecord::create(
    ResetPolarity polarity, ResetTiming assertion, ResetTiming deassertion,
    ResetInitialState initialState, std::optional<ClockDomainRef> synchronousTo,
    std::uint32_t releaseLatencyCycles) {
  if (!isKnown(polarity) || !isKnown(assertion) || !isKnown(deassertion) ||
      !isKnown(initialState))
    return invalidContract("reset", "unknown closed-enum value");

  const bool namesClock = assertion == ResetTiming::Synchronous ||
                          deassertion == ResetTiming::Synchronous;
  if (namesClock != synchronousTo.has_value())
    return invalidContract(
        "reset", namesClock
                     ? "synchronous timing requires synchronous_to"
                     : "fully asynchronous reset forbids synchronous_to");

  if (!synchronousTo && releaseLatencyCycles != 0)
    return invalidContract(
        "reset", "clock-measured release latency requires synchronous_to");

  return ResetDomainContractRecord(polarity, assertion, deassertion,
                                   initialState, std::move(synchronousTo),
                                   releaseLatencyCycles);
}

llvm::Expected<ClockCrossingContractRecord>
ClockCrossingContractRecord::createAsyncFifo(
    FabricTransferPatternRef transferPattern, ClockDomainRef sourceClock,
    ClockDomainRef destinationClock, std::uint32_t depth,
    std::uint32_t synchronizerStages) {
  if (sourceClock == destinationClock)
    return invalidContract("AsyncFifo clock crossing",
                           "source and destination clocks must differ");
  if (depth == 0)
    return invalidContract("AsyncFifo clock crossing",
                           "depth must be positive");
  if (synchronizerStages == 0)
    return invalidContract("AsyncFifo clock crossing",
                           "synchronizer_stages must be positive");
  return ClockCrossingContractRecord(
      std::move(transferPattern), std::move(sourceClock),
      std::move(destinationClock), depth, synchronizerStages);
}

llvm::Expected<std::vector<std::uint8_t>>
loom::fabric::encodeClockDomainContractRecord(
    const ClockDomainContractRecord &record) {
  FabricByteWriter writer;
  writer.field(record.periodFs());
  writer.field(record.phaseFs());
  return writer.take();
}

llvm::Expected<ClockDomainContractRecord>
loom::fabric::decodeClockDomainContractRecord(
    llvm::ArrayRef<std::uint8_t> bytes) {
  FabricByteReader reader(bytes);
  llvm::Expected<std::uint64_t> period = reader.field();
  if (!period)
    return period.takeError();
  llvm::Expected<std::uint64_t> phase = reader.field();
  if (!phase)
    return phase.takeError();
  if (llvm::Error error = requireFinished(reader, "clock domain"))
    return std::move(error);

  llvm::Expected<ClockDomainContractRecord> record =
      ClockDomainContractRecord::create(*period, *phase);
  if (!record)
    return record.takeError();
  if (llvm::Error error = requireCanonical(
          bytes, *record, encodeClockDomainContractRecord, "clock domain"))
    return std::move(error);
  return std::move(*record);
}

llvm::Expected<std::vector<std::uint8_t>>
loom::fabric::encodeResetDomainContractRecord(
    const ResetDomainContractRecord &record) {
  FabricByteWriter writer;
  writer.tag(resetPolarityWireTag(record.polarity()));
  writer.tag(resetTimingWireTag(record.assertion()));
  writer.tag(resetTimingWireTag(record.deassertion()));
  writer.tag(resetInitialStateWireTag(record.initialState()));
  if (record.synchronousTo()) {
    writer.tag(kOptionalClockPresentWireTag);
    encodeFabricRef(writer, *record.synchronousTo());
  } else {
    writer.tag(kOptionalClockAbsentWireTag);
  }
  writer.tag(record.releaseLatencyCycles());
  return writer.take();
}

llvm::Expected<ResetDomainContractRecord>
loom::fabric::decodeResetDomainContractRecord(
    llvm::ArrayRef<std::uint8_t> bytes) {
  FabricByteReader reader(bytes);
  llvm::Expected<ResetPolarity> polarity = readResetPolarity(reader);
  if (!polarity)
    return polarity.takeError();
  llvm::Expected<ResetTiming> assertion =
      readResetTiming(reader, "reset assertion timing");
  if (!assertion)
    return assertion.takeError();
  llvm::Expected<ResetTiming> deassertion =
      readResetTiming(reader, "reset deassertion timing");
  if (!deassertion)
    return deassertion.takeError();
  llvm::Expected<ResetInitialState> initialState =
      readResetInitialState(reader);
  if (!initialState)
    return initialState.takeError();
  llvm::Expected<std::uint32_t> clockTag = reader.tag();
  if (!clockTag)
    return clockTag.takeError();
  if (*clockTag != kOptionalClockAbsentWireTag &&
      *clockTag != kOptionalClockPresentWireTag)
    return malformedRecord("reset synchronous_to",
                           llvm::Twine("unknown discriminant ") +
                               llvm::Twine(*clockTag));

  std::optional<ClockDomainRef> synchronousTo;
  if (*clockTag == kOptionalClockPresentWireTag) {
    ClockDomainRef clock;
    if (llvm::Error error = decodeFabricRefInto(reader, clock))
      return std::move(error);
    synchronousTo = std::move(clock);
  }

  llvm::Expected<std::uint32_t> releaseLatency = reader.tag();
  if (!releaseLatency)
    return releaseLatency.takeError();
  if (llvm::Error error = requireFinished(reader, "reset domain"))
    return std::move(error);

  llvm::Expected<ResetDomainContractRecord> record =
      ResetDomainContractRecord::create(*polarity, *assertion, *deassertion,
                                        *initialState, std::move(synchronousTo),
                                        *releaseLatency);
  if (!record)
    return record.takeError();
  if (llvm::Error error = requireCanonical(
          bytes, *record, encodeResetDomainContractRecord, "reset domain"))
    return std::move(error);
  return std::move(*record);
}

llvm::Expected<std::vector<std::uint8_t>>
loom::fabric::encodeClockCrossingContractRecord(
    const ClockCrossingContractRecord &record) {
  FabricByteWriter writer;
  writer.tag(kAsyncFifoWireTag);
  encodeFabricRef(writer, record.transferPattern());
  encodeFabricRef(writer, record.sourceClock());
  encodeFabricRef(writer, record.destinationClock());
  writer.tag(record.depth());
  writer.tag(record.synchronizerStages());
  return writer.take();
}

llvm::Expected<ClockCrossingContractRecord>
loom::fabric::decodeClockCrossingContractRecord(
    llvm::ArrayRef<std::uint8_t> bytes) {
  FabricByteReader reader(bytes);
  if (llvm::Error error = readAsyncFifoVariant(reader))
    return std::move(error);

  FabricTransferPatternRef transferPattern;
  ClockDomainRef sourceClock;
  ClockDomainRef destinationClock;
  if (llvm::Error error = decodeFabricRefInto(reader, transferPattern))
    return std::move(error);
  if (llvm::Error error = decodeFabricRefInto(reader, sourceClock))
    return std::move(error);
  if (llvm::Error error = decodeFabricRefInto(reader, destinationClock))
    return std::move(error);
  llvm::Expected<std::uint32_t> depth = reader.tag();
  if (!depth)
    return depth.takeError();
  llvm::Expected<std::uint32_t> synchronizerStages = reader.tag();
  if (!synchronizerStages)
    return synchronizerStages.takeError();
  if (llvm::Error error = requireFinished(reader, "clock crossing"))
    return std::move(error);

  llvm::Expected<ClockCrossingContractRecord> record =
      ClockCrossingContractRecord::createAsyncFifo(
          std::move(transferPattern), std::move(sourceClock),
          std::move(destinationClock), *depth, *synchronizerStages);
  if (!record)
    return record.takeError();
  if (llvm::Error error = requireCanonical(
          bytes, *record, encodeClockCrossingContractRecord, "clock crossing"))
    return std::move(error);
  return std::move(*record);
}
