#ifndef LOOM_HARDWARE_RTL_SPECIALIZATION_H
#define LOOM_HARDWARE_RTL_SPECIALIZATION_H

#include "Fabric/Artifact/FabricArtifact.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "Hardware/RTL/CommonSkeleton.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <system_error>
#include <vector>

namespace loom::hardware::rtl {

/// Closed implementation-only recipe catalog owned by the Fabric-to-RTL
/// candidate generator configuration. These choices may not change any
/// Fabric-visible contract.
enum class BackendRecipeKey : std::uint32_t {
  PortableSystemVerilog = 0,
  SynopsysDesignWare = 1,
  CadenceChipWare = 2,
  AmdXilinx = 3,
  IntelAltera = 4,
};

llvm::StringRef backendRecipeKeyword(BackendRecipeKey recipe);

/// One adopted occurrence-scoped projection of ImplementationFlowConfig.
/// The occurrence remains owned by Fabric and the recipe remains owned by the
/// resolved candidate-generator configuration.
struct FabricOperationRecipeBinding final {
  fabric::FabricFuOccurrenceNodeRef occurrence;
  BackendRecipeKey recipe;
  std::vector<ExternalInputBinding> externalInputs;
};

/// Owned provider material consumed by the eventual HardwareImplementation
/// publisher. The descriptor digest is derived only from these exact bytes.
struct FabricOperationProviderPayload final {
  HardwarePayloadRole role;
  std::string logicalName;
  std::string mediaType;
  std::vector<std::uint8_t> bytes;

  HardwarePayload descriptor() const;
};

struct FabricOperationProviderOutput final {
  std::vector<FabricOperationProviderPayload> payloads;
  std::vector<ActivityPoint> activityPoints;
  std::vector<ExternalImplementationBinding> externalImplementationBindings;
};

/// Every reference, view, StringRef, ArrayRef, operation, and platform pointer
/// in a request is callback-scoped and must not be retained by a provider.
struct FabricOperationProviderRequest final {
  /// Isolated, invocation-owned fragment containing only this abstract leaf
  /// and its generator schema. A provider may mutate this fragment but never
  /// the caller's common skeleton.
  mlir::ModuleOp fragment;
  circt::hw::HWModuleGeneratedOp leaf;
  fabric::FabricFuOccurrenceNodeRef occurrence;
  const fabric::ResolvedFabricOpCapabilityView &capability;
  const ConfigurationABI &configurationAbi;
  BackendRecipeKey recipe;
  const platform::ImplementationPlatform *implementationPlatform = nullptr;
  llvm::StringRef externalImplementationContractRef;
  llvm::ArrayRef<ExternalInputBinding> externalInputs;
};

using FabricOperationProviderCallback =
    llvm::Expected<FabricOperationProviderOutput> (*)(
        FabricOperationProviderRequest request);

struct FabricOperationProviderRegistration final {
  ::fabric::ImplementationFamilyId implementationFamily;
  BackendRecipeKey recipe;
  /// Empty for a self-contained provider. Otherwise this is the exact
  /// provider-owned contract schema that interprets externalInputs.
  std::string externalImplementationContractRef;
  FabricOperationProviderCallback callback = nullptr;
};

struct FabricOperationProviderCoverage final {
  ::fabric::ImplementationFamilyId implementationFamily;
  std::vector<BackendRecipeKey> recipes;
};

/// Invocation-local provider availability. Family semantics come only from
/// the generated Fabric registry; this catalog stores callbacks, not another
/// family or operation inventory.
class FabricOperationProviderRegistry final {
public:
  llvm::Error add(FabricOperationProviderRegistration registration);
  std::vector<FabricOperationProviderCoverage> coverage() const;

private:
  const FabricOperationProviderRegistration *
  find(::fabric::ImplementationFamilyId implementationFamily,
       BackendRecipeKey recipe) const;

  std::vector<FabricOperationProviderRegistration> registrations_;

  friend llvm::Expected<FabricOperationProviderOutput>
  specializeFabricOperationLeaves(
      mlir::ModuleOp, const fabric::FinalizedFabricRoot &,
      const FinalizedConfigurationABI &,
      llvm::ArrayRef<FabricOperationLeafAssociation>,
      llvm::ArrayRef<FabricOperationRecipeBinding>,
      const FabricOperationProviderRegistry &,
      const ExternalImplementationContractCatalog &,
      const platform::ImplementationPlatform *);
};

class FabricOperationProviderUnsupportedError final
    : public llvm::ErrorInfo<FabricOperationProviderUnsupportedError> {
public:
  static char ID;

  FabricOperationProviderUnsupportedError(
      ::fabric::ImplementationFamilyId implementationFamily,
      BackendRecipeKey recipe)
      : implementationFamily_(implementationFamily), recipe_(recipe) {}

  ::fabric::ImplementationFamilyId implementationFamily() const {
    return implementationFamily_;
  }
  BackendRecipeKey recipe() const { return recipe_; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  ::fabric::ImplementationFamilyId implementationFamily_;
  BackendRecipeKey recipe_;
};

/// Preflights the complete occurrence recipe and provider closure, prepares
/// every provider in an isolated fragment, and commits only after every
/// fragment verifies. On error the caller's module is unchanged.
llvm::Expected<FabricOperationProviderOutput> specializeFabricOperationLeaves(
    mlir::ModuleOp module, const fabric::FinalizedFabricRoot &fabric,
    const FinalizedConfigurationABI &configurationAbi,
    llvm::ArrayRef<FabricOperationLeafAssociation> operationLeaves,
    llvm::ArrayRef<FabricOperationRecipeBinding> operationRecipes,
    const FabricOperationProviderRegistry &providers,
    const ExternalImplementationContractCatalog &externalContracts,
    const platform::ImplementationPlatform *implementationPlatform = nullptr);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_SPECIALIZATION_H
