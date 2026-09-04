#ifndef LOOM_HARDWARE_IMPLEMENTATION_REPRESENTATIONINDEX_H
#define LOOM_HARDWARE_IMPLEMENTATION_REPRESENTATIONINDEX_H

#include "Common/BlobStore.h"
#include "Hardware/Implementation/ImplementationPayload.h"
#include "Hardware/Implementation/RepresentationLocator.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::hardware {

struct ImplementationRepresentationRoot;

struct ImplementationPayloadBytes final {
  PayloadRole role;
  llvm::StringRef canonicalLogicalName;
  llvm::ArrayRef<std::uint8_t> contents;
};

enum class RepresentationSignalDirection : std::uint32_t {
  Input = 0,
  Output = 1,
  Inout = 2,
};

struct RepresentationSignalGeometry final {
  RepresentationSignalDirection direction;
  std::uint64_t bitWidth;

  friend bool operator==(const RepresentationSignalGeometry &lhs,
                         const RepresentationSignalGeometry &rhs) {
    return lhs.direction == rhs.direction && lhs.bitWidth == rhs.bitWidth;
  }
};

struct RepresentationObjectFacts final {
  RepresentationObjectKind objectKind;
  std::optional<RepresentationSignalGeometry> signalGeometry;

  friend bool operator==(const RepresentationObjectFacts &lhs,
                         const RepresentationObjectFacts &rhs) {
    return lhs.objectKind == rhs.objectKind &&
           lhs.signalGeometry == rhs.signalGeometry;
  }
};

struct RepresentationBoundaryPort final {
  RepresentationLocator locator;
  RepresentationSignalGeometry geometry;

  friend bool operator==(const RepresentationBoundaryPort &lhs,
                         const RepresentationBoundaryPort &rhs) {
    return lhs.locator == rhs.locator && lhs.geometry == rhs.geometry;
  }
};

struct RepresentationModuleInstanceBinding final {
  RepresentationLocator instance;
  RepresentationLocator definition;
};

enum class RepresentationIndexFailureKind : std::uint32_t {
  Invalid = 0,
  Unsupported = 1,
};

class RepresentationIndexFailure final
    : public llvm::ErrorInfo<RepresentationIndexFailure> {
public:
  static char ID;

  RepresentationIndexFailure(RepresentationIndexFailureKind kind,
                             std::string reason)
      : kind_(kind), reason_(std::move(reason)) {}

  RepresentationIndexFailureKind kind() const { return kind_; }
  llvm::StringRef reason() const { return reason_; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  RepresentationIndexFailureKind kind_;
  std::string reason_;
};

class RepresentationIndex final {
public:
  RepresentationFormatDescriptorRef formatRef() const { return formatRef_; }
  RepresentationRootVariant rootVariant() const { return rootVariant_; }
  std::optional<RepresentationPhysicalStage> stage() const { return stage_; }
  const RepresentationLocator &exactRoot() const { return exactRoot_; }

  llvm::Expected<std::optional<RepresentationObjectFacts>>
  lookup(const RepresentationLocator &locator) const;

  std::vector<RepresentationBoundaryPort> rootBoundaryPorts() const;

  /// Concrete module definitions in the complete admitted HDL payload closure.
  /// Non-HDL representations have no module definitions or instance bindings.
  llvm::ArrayRef<RepresentationLocator> concreteModuleDefinitions() const {
    return concreteModuleDefinitions_;
  }

  /// Direct module instances of the exact HDL root, including instances in
  /// named generate scopes. A nested module body is a separate boundary.
  llvm::ArrayRef<RepresentationModuleInstanceBinding>
  rootModuleInstanceBindings() const {
    return rootModuleInstanceBindings_;
  }

  llvm::ArrayRef<RepresentationLocator> unresolvedExternalDefinitions() const {
    return unresolvedExternalDefinitions_;
  }

private:
  struct Entry final {
    RepresentationLocator locator;
    RepresentationObjectFacts facts;
  };

  RepresentationIndex(
      RepresentationFormatDescriptorRef formatRef,
      RepresentationRootVariant rootVariant,
      std::optional<RepresentationPhysicalStage> stage,
      RepresentationLocator exactRoot, std::vector<Entry> entries,
      std::vector<RepresentationLocator> unresolved,
      std::vector<RepresentationLocator> definitions,
      std::vector<RepresentationModuleInstanceBinding> instances)
      : formatRef_(formatRef), rootVariant_(rootVariant), stage_(stage),
        exactRoot_(std::move(exactRoot)), entries_(std::move(entries)),
        unresolvedExternalDefinitions_(std::move(unresolved)),
        concreteModuleDefinitions_(std::move(definitions)),
        rootModuleInstanceBindings_(std::move(instances)) {}

  RepresentationFormatDescriptorRef formatRef_;
  RepresentationRootVariant rootVariant_;
  std::optional<RepresentationPhysicalStage> stage_;
  RepresentationLocator exactRoot_;
  std::vector<Entry> entries_;
  std::vector<RepresentationLocator> unresolvedExternalDefinitions_;
  std::vector<RepresentationLocator> concreteModuleDefinitions_;
  std::vector<RepresentationModuleInstanceBinding> rootModuleInstanceBindings_;

  friend llvm::Expected<RepresentationIndex>
  indexRepresentation(RepresentationFormatDescriptorRef,
                      const RepresentationLocator &,
                      llvm::ArrayRef<ImplementationPayload>, const BlobStore &);
  friend llvm::Expected<RepresentationIndex>
  indexProspectiveRepresentation(RepresentationFormatDescriptorRef,
                                 const RepresentationLocator &,
                                 llvm::ArrayRef<ImplementationPayloadBytes>);
};

/// Purely indexes one canonical payload closure through the selected static
/// descriptor. No path, parser option, or execution authority is accepted.
llvm::Expected<RepresentationIndex>
indexRepresentation(RepresentationFormatDescriptorRef formatRef,
                    const RepresentationLocator &exactRoot,
                    llvm::ArrayRef<ImplementationPayload> canonicalPayloads,
                    const BlobStore &blobs);

/// Indexes one prospective HDL payload closure without first publishing its
/// bytes. The same static descriptor and parser own stored and prospective
/// payload forms.
llvm::Expected<RepresentationIndex> indexProspectiveRepresentation(
    RepresentationFormatDescriptorRef formatRef,
    const RepresentationLocator &exactRoot,
    llvm::ArrayRef<ImplementationPayloadBytes> payloads);

/// Indexes and verifies one complete typed representation root, including its
/// exact variant and physical stage claim.
llvm::Expected<RepresentationIndex>
indexRepresentationRoot(const ImplementationRepresentationRoot &root,
                        const BlobStore &blobs);

} // namespace loom::hardware

#endif // LOOM_HARDWARE_IMPLEMENTATION_REPRESENTATIONINDEX_H
