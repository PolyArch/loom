#ifndef LOOM_HARDWARE_CONFIGURATION_CONFIGURATIONABI_H
#define LOOM_HARDWARE_CONFIGURATION_CONFIGURATIONABI_H

#include "Common/Artifact.h"
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
    "loom.configuration_abi", SchemaVersion{1, 0}};

using ProgrammingUnitId = std::uint64_t;

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
  fabric::FabricSemanticConfigFieldRef field;
  SemanticFieldEncoding semanticEncoding;
  std::vector<DestinationSlice> destinationSlices;
  std::vector<std::uint8_t> inactiveValue;
};

struct ProgrammingUnitDraft final {
  std::vector<fabric::FabricInventoryOwnerRef> exactFabricResourceClosure;
  std::uint64_t payloadBitCount = 0;
  std::vector<ConfigurationFieldEncoding> fields;
};

struct ConfigurationABIDraft final {
  ArtifactRootReference fabric;
  std::vector<ProgrammingUnitDraft> programmingUnits;
};

struct ProgrammingUnit final {
  ProgrammingUnitId id = 0;
  std::vector<fabric::FabricInventoryOwnerRef> exactFabricResourceClosure;
  std::uint64_t payloadBitCount = 0;
  std::vector<ConfigurationFieldEncoding> fields;
};

struct SemanticConfigurationValue final {
  fabric::FabricSemanticConfigFieldRef field;
  std::vector<std::uint8_t> value;
};

class ConfigurationABI final {
public:
  const ArtifactRootReference &fabric() const { return fabric_; }
  llvm::ArrayRef<ProgrammingUnit> programmingUnits() const {
    return programmingUnits_;
  }
  const ProgrammingUnit *findProgrammingUnit(ProgrammingUnitId id) const;

  llvm::Expected<std::vector<std::uint8_t>>
  encode(ProgrammingUnitId id,
         llvm::ArrayRef<SemanticConfigurationValue> values) const;

  llvm::Expected<std::vector<SemanticConfigurationValue>>
  decode(ProgrammingUnitId id, llvm::ArrayRef<std::uint8_t> payload) const;

private:
  ConfigurationABI(ArtifactRootReference fabric,
                   std::vector<ProgrammingUnit> programmingUnits)
      : fabric_(std::move(fabric)),
        programmingUnits_(std::move(programmingUnits)) {}

  ArtifactRootReference fabric_;
  std::vector<ProgrammingUnit> programmingUnits_;

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
