#ifndef LOOM_FABRIC_IDENTITY_FABRICTEMPORALPECONFIGURATION_H
#define LOOM_FABRIC_IDENTITY_FABRICTEMPORALPECONFIGURATION_H

#include "Common/Artifact.h"
#include "Fabric/IR/FabricEnums.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace loom::fabric {

class FabricArtifactView;

enum class FabricTemporalPeSelectorKind : std::uint32_t {
  Route,
  Discard,
  Disconnected,
};

struct FabricTemporalPePortTarget final {
  FabricOrdinal ordinal = 0;

  friend bool operator==(FabricTemporalPePortTarget lhs,
                         FabricTemporalPePortTarget rhs) {
    return lhs.ordinal == rhs.ordinal;
  }
};

struct FabricTemporalPeRegisterFifoTarget final {
  FabricOrdinal ordinal = 0;

  friend bool operator==(FabricTemporalPeRegisterFifoTarget lhs,
                         FabricTemporalPeRegisterFifoTarget rhs) {
    return lhs.ordinal == rhs.ordinal;
  }
};

using FabricTemporalPeSelectorTarget =
    std::variant<FabricTemporalPePortTarget,
                 FabricTemporalPeRegisterFifoTarget>;

struct FabricTemporalPeOperandSelection final {
  FabricTemporalPeSelectorKind kind =
      FabricTemporalPeSelectorKind::Disconnected;
  std::optional<FabricTemporalPeSelectorTarget> target;
  llvm::APInt tag = llvm::APInt(1, 0);

  friend bool operator==(const FabricTemporalPeOperandSelection &lhs,
                         const FabricTemporalPeOperandSelection &rhs) {
    return lhs.kind == rhs.kind && lhs.target == rhs.target &&
           lhs.tag == rhs.tag;
  }
};

struct FabricTemporalPeResultSelection final {
  FabricTemporalPeSelectorKind kind =
      FabricTemporalPeSelectorKind::Disconnected;
  std::optional<FabricTemporalPeSelectorTarget> target;
  llvm::APInt tag = llvm::APInt(1, 0);

  friend bool operator==(const FabricTemporalPeResultSelection &lhs,
                         const FabricTemporalPeResultSelection &rhs) {
    return lhs.kind == rhs.kind && lhs.target == rhs.target &&
           lhs.tag == rhs.tag;
  }
};

struct FabricTemporalPeInstructionEntry final {
  FabricFuOccurrenceRef selectedFu;
  std::vector<FabricTemporalPeOperandSelection> operandSelections;
  std::vector<FabricTemporalPeResultSelection> resultSelections;

  friend bool operator==(const FabricTemporalPeInstructionEntry &lhs,
                         const FabricTemporalPeInstructionEntry &rhs) {
    return lhs.selectedFu == rhs.selectedFu &&
           lhs.operandSelections == rhs.operandSelections &&
           lhs.resultSelections == rhs.resultSelections;
  }
};

struct FabricTemporalPeDisabled final {
  friend bool operator==(FabricTemporalPeDisabled, FabricTemporalPeDisabled) {
    return true;
  }
};

struct FabricTemporalPeActive final {
  std::vector<std::optional<FabricTemporalPeInstructionEntry>> rows;

  friend bool operator==(const FabricTemporalPeActive &lhs,
                         const FabricTemporalPeActive &rhs) {
    return lhs.rows == rhs.rows;
  }
};

using FabricTemporalPeConfigurationValue =
    std::variant<FabricTemporalPeDisabled, FabricTemporalPeActive>;

struct FabricTemporalPeFuShape final {
  FabricFuOccurrenceRef fu;
  std::uint32_t inputCount = 0;
  std::uint32_t outputCount = 0;
};

/// Exact fixed-capacity direct-carrier layout for one Temporal PE. Offsets are
/// little-endian bit offsets within the semantic carrier and are derived only
/// from the immutable Fabric shape.
struct FabricTemporalPeConfigurationLayout final {
  std::uint32_t contextCount = 0;
  std::uint32_t inputPortCount = 0;
  std::uint32_t outputPortCount = 0;
  std::uint32_t registerFifoCount = 0;
  std::uint32_t tagWidthBits = 0;
  std::uint32_t selectedFuBitCount = 0;
  std::uint32_t inputTargetBitCount = 0;
  std::uint32_t outputTargetBitCount = 0;
  std::uint32_t operandSelectionBitCount = 0;
  std::uint32_t resultSelectionBitCount = 0;
  std::uint32_t maximumFuInputCount = 0;
  std::uint32_t maximumFuOutputCount = 0;
  std::uint64_t rowBitCount = 0;
  std::uint64_t carrierBitCount = 0;
  std::vector<FabricTemporalPeFuShape> fus;

  std::uint64_t rowOffset(std::uint32_t context) const;
  std::uint64_t selectedFuOffset(std::uint32_t context) const;
  std::uint64_t operandSelectionOffset(std::uint32_t context,
                                       std::uint32_t input) const;
  std::uint64_t resultSelectionOffset(std::uint32_t context,
                                      std::uint32_t output) const;
};

/// Sealed, rebuildable direct relation for one Temporal PE instruction table.
/// FU and operation configuration fields remain separate residency-qualified
/// slots and are deliberately absent from this carrier.
class FabricTemporalPeConfigurationSchemaView final {
public:
  FabricPeOccurrenceRef pe() const { return pe_; }
  const FabricSemanticConfigFieldRef &field() const { return field_; }
  const FabricTemporalPeConfigurationLayout &layout() const { return layout_; }

  llvm::Expected<CanonicalSemanticBytes>
  encode(const FabricTemporalPeConfigurationValue &value) const;

  llvm::Expected<FabricTemporalPeConfigurationValue>
  decode(llvm::ArrayRef<std::uint8_t> bytes) const;

private:
  FabricTemporalPeConfigurationSchemaView(
      FabricPeOccurrenceRef pe, FabricSemanticConfigFieldRef field,
      FabricTemporalPeConfigurationLayout layout)
      : pe_(pe), field_(std::move(field)), layout_(std::move(layout)) {}

  FabricPeOccurrenceRef pe_;
  FabricSemanticConfigFieldRef field_;
  FabricTemporalPeConfigurationLayout layout_;

  friend class FabricArtifactView;
};

} // namespace loom::fabric

#endif // LOOM_FABRIC_IDENTITY_FABRICTEMPORALPECONFIGURATION_H
