#ifndef LOOM_FABRIC_IDENTITY_FABRICPECONFIGURATION_H
#define LOOM_FABRIC_IDENTITY_FABRICPECONFIGURATION_H

#include "Common/Artifact.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace loom::fabric {

class FabricArtifactView;

enum class FabricPeConfigurationFieldKind : std::uint32_t {
  Activation,
  InputSelector,
  OutputSelector,
};

struct FabricPeDisabled final {
  friend bool operator==(FabricPeDisabled, FabricPeDisabled) { return true; }
};

struct FabricPeActive final {
  FabricFuOccurrenceRef fu;

  friend bool operator==(const FabricPeActive &lhs, const FabricPeActive &rhs) {
    return lhs.fu == rhs.fu;
  }
};

struct FabricPeDisconnected final {
  friend bool operator==(FabricPeDisconnected, FabricPeDisconnected) {
    return true;
  }
};

struct FabricPeRoute final {
  FabricTransportEndpointRef endpoint;

  friend bool operator==(const FabricPeRoute &lhs, const FabricPeRoute &rhs) {
    return lhs.endpoint == rhs.endpoint;
  }
};

struct FabricPeInputDiscard final {
  FabricTransportEndpointRef endpoint;

  friend bool operator==(const FabricPeInputDiscard &lhs,
                         const FabricPeInputDiscard &rhs) {
    return lhs.endpoint == rhs.endpoint;
  }
};

struct FabricPeOutputDiscard final {
  friend bool operator==(FabricPeOutputDiscard, FabricPeOutputDiscard) {
    return true;
  }
};

using FabricPeConfigurationValue =
    std::variant<FabricPeDisabled, FabricPeActive, FabricPeDisconnected,
                 FabricPeRoute, FabricPeInputDiscard, FabricPeOutputDiscard>;

struct FabricPeConfigurationFieldView final {
  FabricSemanticConfigFieldRef reference;
  FabricPeConfigurationFieldKind kind =
      FabricPeConfigurationFieldKind::Activation;
  std::optional<FabricFuOccurrencePortRef> port;

  friend bool operator==(const FabricPeConfigurationFieldView &lhs,
                         const FabricPeConfigurationFieldView &rhs) {
    return lhs.reference == rhs.reference && lhs.kind == rhs.kind &&
           lhs.port == rhs.port;
  }
};

/// A sealed, rebuildable projection of one Spatial PE's static semantic
/// configuration fields. Its finite domains factor activation and selectors;
/// physical codes and packing remain ConfigurationABI-owned.
class FabricSpatialPeConfigurationSchemaView final {
public:
  FabricPeOccurrenceRef pe() const { return pe_; }
  llvm::ArrayRef<FabricPeConfigurationFieldView> fields() const {
    return fields_;
  }
  llvm::ArrayRef<FabricTransportEndpointRef> inputEndpoints() const {
    return inputEndpoints_;
  }
  llvm::ArrayRef<FabricTransportEndpointRef> outputEndpoints() const {
    return outputEndpoints_;
  }

  llvm::Expected<std::vector<FabricPeConfigurationValue>>
  finiteDomain(const FabricSemanticConfigFieldRef &field) const;

  llvm::Expected<CanonicalSemanticBytes>
  encode(const FabricSemanticConfigFieldRef &field,
         const FabricPeConfigurationValue &value) const;

  llvm::Expected<FabricPeConfigurationValue>
  decode(const FabricSemanticConfigFieldRef &field,
         llvm::ArrayRef<std::uint8_t> bytes) const;

private:
  FabricSpatialPeConfigurationSchemaView(
      FabricPeOccurrenceRef pe,
      std::vector<FabricPeConfigurationFieldView> fields,
      std::vector<FabricFuOccurrenceRef> fuOccurrences,
      std::vector<FabricTransportEndpointRef> inputEndpoints,
      std::vector<FabricTransportEndpointRef> outputEndpoints)
      : pe_(pe), fields_(std::move(fields)),
        fuOccurrences_(std::move(fuOccurrences)),
        inputEndpoints_(std::move(inputEndpoints)),
        outputEndpoints_(std::move(outputEndpoints)) {}

  const FabricPeConfigurationFieldView *
  find(const FabricSemanticConfigFieldRef &field) const;

  FabricPeOccurrenceRef pe_;
  std::vector<FabricPeConfigurationFieldView> fields_;
  std::vector<FabricFuOccurrenceRef> fuOccurrences_;
  std::vector<FabricTransportEndpointRef> inputEndpoints_;
  std::vector<FabricTransportEndpointRef> outputEndpoints_;

  friend class FabricArtifactView;
};

} // namespace loom::fabric

#endif // LOOM_FABRIC_IDENTITY_FABRICPECONFIGURATION_H
