#ifndef LOOM_HARDWARE_CONFIGURATION_CONFIGURATIONABI_H
#define LOOM_HARDWARE_CONFIGURATION_CONFIGURATIONABI_H

#include "Common/Artifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

namespace loom {
class ArtifactStore;
}

namespace loom::hardware {

inline constexpr ArtifactSchemaDescriptor configurationAbiSchema{
    "loom.configuration_abi", SchemaVersion{3, 0}};

using ProgrammingUnitId = std::uint64_t;

/// Exact persistent reference to one programming unit in one finalized
/// ConfigurationABI. Its binary codec is owned by this schema.
struct ProgrammingUnitRef final {
  ArtifactRootReference configurationAbi;
  ProgrammingUnitId unitId = 0;

  friend bool operator==(const ProgrammingUnitRef &lhs,
                         const ProgrammingUnitRef &rhs) {
    return lhs.configurationAbi == rhs.configurationAbi &&
           lhs.unitId == rhs.unitId;
  }
  friend bool operator!=(const ProgrammingUnitRef &lhs,
                         const ProgrammingUnitRef &rhs) {
    return !(lhs == rhs);
  }
};

std::vector<std::uint8_t>
encodeProgrammingUnitRef(const ProgrammingUnitRef &reference);

llvm::Expected<ProgrammingUnitRef>
decodeProgrammingUnitRef(llvm::ArrayRef<std::uint8_t> bytes,
                         const ArtifactStore &store);

namespace detail {
llvm::Expected<ProgrammingUnitRef>
decodeProgrammingUnitRefFraming(llvm::ArrayRef<std::uint8_t> bytes);
}

struct DestinationSlice final {
  std::uint64_t sourceBitOffset = 0;
  std::uint64_t destinationBitOffset = 0;
  std::uint64_t bitCount = 0;

  friend bool operator==(const DestinationSlice &lhs,
                         const DestinationSlice &rhs) {
    return lhs.sourceBitOffset == rhs.sourceBitOffset &&
           lhs.destinationBitOffset == rhs.destinationBitOffset &&
           lhs.bitCount == rhs.bitCount;
  }
};

struct DirectBitsEncoding final {
  std::uint64_t encodedBitCount = 0;
};

struct FiniteCodebookEntry final {
  std::vector<std::uint8_t> semanticValue;
  std::vector<std::uint8_t> physicalCode;

  friend bool operator==(const FiniteCodebookEntry &lhs,
                         const FiniteCodebookEntry &rhs) {
    return lhs.semanticValue == rhs.semanticValue &&
           lhs.physicalCode == rhs.physicalCode;
  }
};

struct FiniteCodebookEncoding final {
  std::uint64_t encodedBitCount = 0;
  std::vector<FiniteCodebookEntry> entries;
};

using SemanticFieldEncoding =
    std::variant<DirectBitsEncoding, FiniteCodebookEncoding>;

struct ConfigurationFieldEncoding final {
  fabric::FabricPhysicalConfigurationSlotRef slot;
  SemanticFieldEncoding semanticEncoding;
  std::vector<DestinationSlice> destinationSlices;
  std::vector<std::uint8_t> inactiveValue;

  std::uint64_t encodedBitCount() const {
    return std::visit(
        [](const auto &encoding) { return encoding.encodedBitCount; },
        semanticEncoding);
  }
};

struct ProgrammingUnitDraft final {
  std::vector<fabric::FabricPhysicalOccurrenceOwnerRef>
      exactFabricResourceClosure;
  std::uint64_t payloadBitCount = 0;
  std::vector<ConfigurationFieldEncoding> fields;
};

struct ConfigurationABIDraft final {
  ArtifactRootReference fabric;
  std::vector<ProgrammingUnitDraft> programmingUnits;
};

struct ProgrammingUnit final {
  ProgrammingUnitId id = 0;
  std::vector<fabric::FabricPhysicalOccurrenceOwnerRef>
      exactFabricResourceClosure;
  std::uint64_t payloadBitCount = 0;
  std::vector<ConfigurationFieldEncoding> fields;
};

/// Removable projection of one programming unit's occurrence ownership.
/// Direct System resources and imported SpatialCore occurrences remain
/// distinct because only the latter can be implemented by a local RTL root.
struct ProgrammingUnitOccurrenceScope final {
  bool includesDirectSystemResources = false;
  std::vector<fabric::SpatialCoreOccurrenceRef> spatialCores;
};

ProgrammingUnitOccurrenceScope
deriveProgrammingUnitOccurrenceScope(const ProgrammingUnit &unit);

struct SemanticConfigurationValue final {
  fabric::FabricPhysicalConfigurationSlotRef slot;
  std::vector<std::uint8_t> value;
};

class ConfigurationABI final {
public:
  const ArtifactRootReference &fabric() const { return fabric_; }
  const fabric::FabricSystemRootView &fabricSystem() const { return system_; }
  llvm::ArrayRef<ProgrammingUnit> programmingUnits() const {
    return programmingUnits_;
  }
  const ProgrammingUnit *findProgrammingUnit(ProgrammingUnitId id) const;
  const ConfigurationFieldEncoding *
  findField(const fabric::FabricPhysicalConfigurationSlotRef &slot) const;
  const ConfigurationFieldEncoding *
  findOperationField(const fabric::FabricPhysicalOccurrenceOwnerRef &operation,
                     fabric::FabricOrdinal fieldOrdinal) const;

  llvm::Expected<std::vector<std::uint8_t>>
  encode(ProgrammingUnitId id,
         llvm::ArrayRef<SemanticConfigurationValue> values) const;

  llvm::Expected<std::vector<SemanticConfigurationValue>>
  decode(ProgrammingUnitId id, llvm::ArrayRef<std::uint8_t> payload) const;

private:
  ConfigurationABI(ArtifactRootReference fabric,
                   std::vector<ProgrammingUnit> programmingUnits,
                   fabric::FabricSystemRootView system)
      : fabric_(std::move(fabric)),
        programmingUnits_(std::move(programmingUnits)),
        system_(std::move(system)) {}

  ArtifactRootReference fabric_;
  std::vector<ProgrammingUnit> programmingUnits_;
  fabric::FabricSystemRootView system_;

  friend llvm::Expected<class FinalizedConfigurationABI>
  finalizeConfigurationABI(ConfigurationABIDraft, const ArtifactStore &);
  friend llvm::Expected<class FinalizedConfigurationABI>
  importConfigurationABI(const ArtifactRootReference &, const ArtifactStore &);
};

class FinalizedConfigurationABI final {
public:
  const ArtifactRootReference &reference() const { return reference_; }
  const CanonicalSemanticBytes &canonicalBytes() const {
    return canonicalBytes_;
  }
  const ConfigurationABI &abi() const { return abi_; }

private:
  FinalizedConfigurationABI(ArtifactRootReference reference,
                            CanonicalSemanticBytes canonicalBytes,
                            ConfigurationABI abi)
      : reference_(std::move(reference)),
        canonicalBytes_(std::move(canonicalBytes)), abi_(std::move(abi)) {}

  ArtifactRootReference reference_;
  CanonicalSemanticBytes canonicalBytes_;
  ConfigurationABI abi_;

  friend llvm::Expected<FinalizedConfigurationABI>
  finalizeConfigurationABI(ConfigurationABIDraft, const ArtifactStore &);
  friend llvm::Expected<FinalizedConfigurationABI>
  importConfigurationABI(const ArtifactRootReference &, const ArtifactStore &);
};

llvm::Expected<FinalizedConfigurationABI>
finalizeConfigurationABI(ConfigurationABIDraft draft,
                         const ArtifactStore &store);

llvm::Expected<FinalizedConfigurationABI>
importConfigurationABI(const ArtifactRootReference &reference,
                       const ArtifactStore &store);

} // namespace loom::hardware

#endif // LOOM_HARDWARE_CONFIGURATION_CONFIGURATIONABI_H
