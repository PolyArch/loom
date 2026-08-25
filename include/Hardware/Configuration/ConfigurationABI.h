#ifndef LOOM_HARDWARE_CONFIGURATION_CONFIGURATIONABI_H
#define LOOM_HARDWARE_CONFIGURATION_CONFIGURATIONABI_H

#include "Common/Artifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

namespace loom {
class ArtifactStore;
}

namespace loom::hardware {

namespace detail {
class ConfigurationABIImportSessionState;
}

enum class ConfigurationABIImportSessionMode : std::uint8_t {
  ReuseEnclosing,
  Isolated,
};

struct ConfigurationABIImportSessionStatistics final {
  std::uint64_t importRequests = 0;
  std::uint64_t uniqueConstructions = 0;
  std::uint64_t cacheHits = 0;
  std::uint64_t cacheMisses = 0;
  std::uint64_t bytesRead = 0;
  std::uint64_t bytesCopied = 0;
  std::uint64_t constructionNanoseconds = 0;
  std::uint64_t deterministicWork = 0;
  std::uint64_t retainedBytes = 0;
  std::uint64_t entryCount = 0;
};

/// Owns immutable strict ConfigurationABI imports for one synchronous
/// verification domain. An isolated domain is used by independent replay.
class ConfigurationABIImportSession final {
public:
  explicit ConfigurationABIImportSession(
      ConfigurationABIImportSessionMode mode =
          ConfigurationABIImportSessionMode::ReuseEnclosing);
  ~ConfigurationABIImportSession();

  ConfigurationABIImportSession(const ConfigurationABIImportSession &) =
      delete;
  ConfigurationABIImportSession &
  operator=(const ConfigurationABIImportSession &) = delete;

  ConfigurationABIImportSessionStatistics statistics() const;

private:
  std::unique_ptr<detail::ConfigurationABIImportSessionState> owned_;
  detail::ConfigurationABIImportSessionState *active_ = nullptr;
  detail::ConfigurationABIImportSessionState *previous_ = nullptr;
};

inline constexpr ArtifactSchemaDescriptor configurationAbiSchema{
    "loom.configuration_abi", SchemaVersion{4, 0}};

using ProgrammingUnitId = std::uint64_t;
using ConfigurationEncodingRelationId = std::uint64_t;

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

struct ConfigurationEncodingRelationDraft final {
  SemanticFieldEncoding semanticEncoding;
  std::vector<std::uint8_t> inactiveValue;

  std::uint64_t encodedBitCount() const {
    return std::visit(
        [](const auto &encoding) { return encoding.encodedBitCount; },
        semanticEncoding);
  }
};

struct ConfigurationEncodingRelation final {
  ConfigurationEncodingRelationId id = 0;
  SemanticFieldEncoding semanticEncoding;
  std::vector<std::uint8_t> inactiveValue;

  std::uint64_t encodedBitCount() const {
    return std::visit(
        [](const auto &encoding) { return encoding.encodedBitCount; },
        semanticEncoding);
  }
};

struct ConfigurationFieldEncoding final {
  fabric::FabricPhysicalConfigurationSlotRef slot;
  ConfigurationEncodingRelationId encodingRelation = 0;
  std::vector<DestinationSlice> destinationSlices;
};

struct ProgrammingUnitDraft final {
  std::vector<fabric::FabricPhysicalOccurrenceOwnerRef>
      exactFabricResourceClosure;
  std::uint64_t payloadBitCount = 0;
  std::vector<ConfigurationFieldEncoding> fields;
};

struct ConfigurationABIDraft final {
  ArtifactRootReference fabric;
  std::vector<ConfigurationEncodingRelationDraft> encodingRelations;
  std::vector<ProgrammingUnitDraft> programmingUnits;
};

struct ConfigurationABIConstructionStatistics final {
  std::uint64_t canonicalizationCount = 0;
  std::uint64_t canonicalizationNanoseconds = 0;
  std::uint64_t semanticValidationCacheHits = 0;
  std::uint64_t semanticValidationCacheMisses = 0;
  std::uint64_t physicalSlotValidationCount = 0;
  std::uint64_t retainedCacheBytes = 0;
  std::uint64_t deterministicWork = 0;
  std::uint64_t encodingRelationCount = 0;
  std::uint64_t configurationFieldCount = 0;
  std::uint64_t canonicalByteCount = 0;
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
  llvm::ArrayRef<ConfigurationEncodingRelation> encodingRelations() const {
    return encodingRelations_;
  }
  llvm::ArrayRef<ProgrammingUnit> programmingUnits() const {
    return programmingUnits_;
  }
  const ConfigurationEncodingRelation *
  findEncodingRelation(ConfigurationEncodingRelationId id) const;
  const ConfigurationEncodingRelation *
  findEncodingRelation(const ConfigurationFieldEncoding &field) const {
    return findEncodingRelation(field.encodingRelation);
  }
  const ProgrammingUnit *findProgrammingUnit(ProgrammingUnitId id) const;
  const ConfigurationFieldEncoding *
  findField(const fabric::FabricPhysicalConfigurationSlotRef &slot) const;
  const ConfigurationFieldEncoding *findField(
      ProgrammingUnitId unit,
      const fabric::FabricPhysicalConfigurationSlotRef &slot) const;
  const ConfigurationFieldEncoding *
  findOperationField(const fabric::FabricPhysicalOccurrenceOwnerRef &operation,
                     fabric::FabricOrdinal fieldOrdinal) const;
  const ConfigurationEncodingRelation *findOperationEncodingRelation(
      const fabric::FabricPhysicalOccurrenceOwnerRef &operation,
      fabric::FabricOrdinal fieldOrdinal) const;

  llvm::Expected<std::vector<std::uint8_t>>
  encode(ProgrammingUnitId id,
         llvm::ArrayRef<SemanticConfigurationValue> values) const;

  llvm::Expected<std::vector<SemanticConfigurationValue>>
  decode(ProgrammingUnitId id, llvm::ArrayRef<std::uint8_t> payload) const;

private:
  ConfigurationABI(ArtifactRootReference fabric,
                   std::vector<ConfigurationEncodingRelation> encodingRelations,
                   std::vector<ProgrammingUnit> programmingUnits,
                   fabric::FabricSystemRootView system)
      : fabric_(std::move(fabric)),
        encodingRelations_(std::move(encodingRelations)),
        programmingUnits_(std::move(programmingUnits)),
        system_(std::move(system)) {}

  ArtifactRootReference fabric_;
  std::vector<ConfigurationEncodingRelation> encodingRelations_;
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
  const ConfigurationABI &abi() const { return *abi_; }
  const ConfigurationABIConstructionStatistics &constructionStatistics() const {
    return constructionStatistics_;
  }

private:
  FinalizedConfigurationABI(ArtifactRootReference reference,
                            CanonicalSemanticBytes canonicalBytes,
                            std::shared_ptr<const ConfigurationABI> abi,
                            ConfigurationABIConstructionStatistics statistics)
      : reference_(std::move(reference)),
        canonicalBytes_(std::move(canonicalBytes)), abi_(std::move(abi)),
        constructionStatistics_(statistics) {}

  ArtifactRootReference reference_;
  CanonicalSemanticBytes canonicalBytes_;
  std::shared_ptr<const ConfigurationABI> abi_;
  ConfigurationABIConstructionStatistics constructionStatistics_;

  friend llvm::Expected<FinalizedConfigurationABI>
  finalizeConfigurationABI(ConfigurationABIDraft, const ArtifactStore &);
  friend llvm::Expected<FinalizedConfigurationABI>
  importConfigurationABI(const ArtifactRootReference &, const ArtifactStore &);
};

llvm::Expected<FinalizedConfigurationABI>
finalizeConfigurationABI(ConfigurationABIDraft draft,
                         const ArtifactStore &store);

/// Canonicalizes one draft through the same path as finalization and derives
/// its exact artifact reference without publishing it.
llvm::Expected<ArtifactRootReference>
deriveConfigurationABIArtifactReference(ConfigurationABIDraft draft,
                                        const ArtifactStore &store);

llvm::Expected<FinalizedConfigurationABI>
importConfigurationABI(const ArtifactRootReference &reference,
                       const ArtifactStore &store);

} // namespace loom::hardware

#endif // LOOM_HARDWARE_CONFIGURATION_CONFIGURATIONABI_H
