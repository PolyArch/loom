#ifndef LOOM_LIB_ADG_BUILDERINTERNAL_H
#define LOOM_LIB_ADG_BUILDERINTERNAL_H

#include "ADG/Builder.h"
#include "Fabric/IR/FabricOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::adg::detail {

struct SpatialRootState final {
  ::fabric::ModuleOp operation;
  std::string label;
  std::vector<mlir::Type> resultTypes;
  std::vector<mlir::Operation *> unresolvedBackedges;
  bool closed = false;
};

struct PeState final {
  ::fabric::PeOp operation;
  std::size_t rootOrdinal = 0;
  bool closed = false;
};

struct FuCapabilityTemplateDraft final {
  std::vector<mlir::Operation *> activeOperations;
  std::vector<std::pair<mlir::Operation *, std::uint32_t>> routes;
};

struct FuState final {
  ::fabric::FuOp operation;
  std::size_t rootOrdinal = 0;
  std::size_t peOrdinal = 0;
  bool closed = false;
  std::vector<mlir::Operation *> unresolvedBackedges;
  std::vector<FuCapabilityTemplateDraft> capabilityTemplates;
};

struct ImportedModuleBoundary final {
  loom::fabric::FabricSpatialAttachmentEndpointRef::Plane plane =
      loom::fabric::FabricSpatialAttachmentEndpointRef::Plane::Transport;
  loom::fabric::FabricOrdinal occurrenceOrdinal = 0;
};

struct ImportedModuleState final {
  ImportedModuleState(ArtifactRootReference reference,
                      loom::fabric::FabricModuleTemplateRef module)
      : reference(std::move(reference)), module(module) {}

  ArtifactRootReference reference;
  loom::fabric::FabricModuleTemplateRef module;
  std::vector<ImportedModuleBoundary> inputs;
  std::vector<ImportedModuleBoundary> outputs;
  std::uint64_t transportInputCount = 0;
  std::uint64_t transportOutputCount = 0;
  std::uint64_t memoryInputCount = 0;
  std::uint64_t memoryOutputCount = 0;
};

struct SystemEntityState final {
  SystemEntityState(loom::fabric::FabricEntityKind kind,
                    mlir::Operation *operation)
      : kind(kind), operation(operation) {}

  loom::fabric::FabricEntityKind kind =
      loom::fabric::FabricEntityKind::HostCoreOccurrence;
  mlir::Operation *operation = nullptr;
  std::optional<std::size_t> importedModule;
  std::uint64_t inputCount = 0;
  std::uint64_t outputCount = 0;
  std::uint64_t nextTransferPatternOrdinal = 0;
  std::optional<loom::fabric::CanonicalServiceEndpointPlane> endpointPlane;
  std::optional<loom::fabric::CanonicalServiceEndpointRole> endpointRole;
  bool crossingDeclared = false;
  bool closed = true;
};

struct SystemRootState final {
  SystemRootState(::fabric::SystemOp operation, std::string label)
      : operation(operation), label(std::move(label)) {}

  ::fabric::SystemOp operation;
  std::string label;
  std::vector<ImportedModuleState> importedModules;
  std::vector<SystemEntityState> entities;
  bool closed = false;
};

class DesignState final {
public:
  explicit DesignState(const loom::ArtifactStore &store);

  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> draft;
  const loom::ArtifactStore &store;
  std::vector<SpatialRootState> spatialRoots;
  std::vector<SystemRootState> systemRoots;
  std::vector<PeState> pes;
  std::vector<FuState> fus;
  llvm::StringSet<> labels;
  bool consumed = false;
};

llvm::Error invalid(const llvm::Twine &message);

llvm::Expected<std::shared_ptr<DesignState>>
activeState(const std::weak_ptr<DesignState> &weak);

mlir::Type materializePortType(mlir::MLIRContext &context,
                               const PortType &type);

} // namespace loom::adg::detail

#endif // LOOM_LIB_ADG_BUILDERINTERNAL_H
