#ifndef LOOM_LIB_ADG_BUILDERINTERNAL_H
#define LOOM_LIB_ADG_BUILDERINTERNAL_H

#include "ADG/Builder.h"
#include "Fabric/IR/FabricOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/SmallVector.h"
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

class DesignIdentity final {};

struct PeMaterialization final {
  llvm::SmallVector<mlir::Type, 8> boundaryInputTypes;
  llvm::SmallVector<mlir::Type, 8> bodyInputTypes;
  llvm::SmallVector<mlir::Type, 8> outputTypes;
  mlir::IntegerAttr tagWidth;
  mlir::IntegerAttr instructionCapacity;
  mlir::IntegerAttr registerFifoCount;
  mlir::IntegerAttr registerFifoDepth;
  mlir::IntegerAttr registerFifoPorts;
  ::fabric::FuConfigModeAttr fuConfigurationMode;
  ::fabric::OperandBufferModeAttr operandBufferMode;
  mlir::IntegerAttr operandBufferSize;
  std::size_t instructionContexts = 0;
};

struct SwitchMaterialization final {
  llvm::SmallVector<mlir::Type, 8> inputTypes;
  llvm::SmallVector<mlir::Type, 8> outputTypes;
  mlir::ArrayAttr hardwareParameters;
};

struct MemoryMaterialization final {
  llvm::SmallVector<mlir::Type, 8> inputTypes;
  llvm::SmallVector<mlir::Type, 8> outputTypes;
  ::fabric::MemoryContractAttr contract;
  mlir::ArrayAttr operationPorts;
  std::size_t operationPortCount = 0;
  bool hasLocalService = false;
};

/// Canonical translation from public ADG specifications to Fabric types and
/// attributes. Anonymous operations and named templates consume the same
/// result, so declaration form cannot change the hardware contract.
class BuilderSpecMaterializer final {
public:
  static bool samePortKind(mlir::Type left, mlir::Type right);

  static llvm::Expected<PeMaterialization>
  pe(mlir::MLIRContext &context, llvm::ArrayRef<mlir::Type> boundaryInputTypes,
     const PeSpec &spec, bool namedTemplate);

  static llvm::Expected<SwitchMaterialization>
  switchSpec(mlir::MLIRContext &context, const SwitchSpec &spec);

  static llvm::Expected<MemoryMaterialization>
  memory(mlir::MLIRContext &context, const MemorySpec &spec);
};

struct SpatialRootState final {
  ::fabric::ModuleOp operation;
  std::string label;
  std::vector<mlir::Type> resultTypes;
  std::vector<mlir::Operation *> unresolvedBackedges;
  ::fabric::ModuleDomainAuthoringRelation domainRelation;
  bool closed = false;
  std::optional<loom::fabric::FabricArtifactView> derivedParent;
  std::vector<mlir::Value> derivedOutputs;
};

struct PeState final {
  ::fabric::PeOp operation;
  std::size_t rootOrdinal = 0;
  bool named = false;
  bool closed = false;
};

struct FuCapabilityTemplateDraft final {
  std::vector<mlir::Operation *> activeOperations;
  std::vector<std::pair<mlir::Operation *, std::uint32_t>> routes;
  std::optional<loom::fabric::FabricOrdinal> canonicalOrdinal;
  bool handleExposed = false;
};

struct FuState final {
  ::fabric::FuOp operation;
  std::size_t rootOrdinal = 0;
  std::size_t peOrdinal = 0;
  bool named = false;
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
  std::optional<loom::fabric::FabricArtifactView> derivedParent;
  std::vector<ArtifactRootReference> admissibleModules;
};

class DesignState final {
public:
  explicit DesignState(const loom::ArtifactStore &store);

  mlir::MLIRContext context;
  std::shared_ptr<DesignIdentity> identity = std::make_shared<DesignIdentity>();
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

/// Validates that one domain authoring handle belongs to the exact open
/// root. Handles resolve their owning design and root ordinal, nothing more.
llvm::Error checkDomainHandleOwner(const std::shared_ptr<DesignState> &state,
                                   std::size_t rootOrdinal,
                                   const std::weak_ptr<DesignState> &owner,
                                   std::size_t handleRootOrdinal,
                                   llvm::StringRef description);

mlir::Type materializePortType(mlir::MLIRContext &context,
                               const PortType &type);

llvm::Expected<mlir::ModuleOp>
loadCanonicalFabricModule(const loom::fabric::FinalizedFabricRoot &parent,
                          DesignState &state,
                          loom::fabric::FabricRootKind expectedKind);

} // namespace loom::adg::detail

#endif // LOOM_LIB_ADG_BUILDERINTERNAL_H
