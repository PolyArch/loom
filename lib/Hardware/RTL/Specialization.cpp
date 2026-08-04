#include "Hardware/RTL/Specialization.h"

#include "Fabric/Artifact/FabricArtifactLocalReference.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace loom::hardware::rtl {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "rtl_operation_specialization_invalid: " +
                                     message);
}

bool validFamily(::fabric::ImplementationFamilyId family) {
  return static_cast<std::uint32_t>(family) <
         ::fabric::implementationFamilyCount();
}

bool validRecipe(BackendRecipeKey recipe) {
  switch (recipe) {
  case BackendRecipeKey::PortableSystemVerilog:
  case BackendRecipeKey::SynopsysDesignWare:
  case BackendRecipeKey::CadenceChipWare:
  case BackendRecipeKey::AmdXilinx:
  case BackendRecipeKey::IntelAltera:
    return true;
  }
  return false;
}

auto registrationKey(const FabricOperationProviderRegistration &registration) {
  return std::make_tuple(
      static_cast<std::uint32_t>(registration.implementationFamily),
      static_cast<std::uint32_t>(registration.recipe));
}

struct SpecializationJob final {
  std::vector<std::uint8_t> occurrenceKey;
  circt::hw::HWModuleGeneratedOp leaf;
  std::string leafSymbol;
  fabric::FabricFuOccurrenceNodeRef occurrence;
  const fabric::ResolvedFabricOpCapabilityView *capability = nullptr;
  const FabricOperationProviderRegistration *provider = nullptr;
  std::vector<ExternalInputBinding> externalInputs;
  BackendRecipeKey recipe = BackendRecipeKey::PortableSystemVerilog;
};

struct PreparedSpecialization final {
  SpecializationJob *job = nullptr;
  mlir::OwningOpRef<mlir::ModuleOp> fragment;
  FabricOperationProviderOutput output;
};

bool dependencyLess(const ExternalDependencyIdentity &lhs,
                    const ExternalDependencyIdentity &rhs) {
  if (lhs.index() != rhs.index())
    return lhs.index() < rhs.index();
  if (const auto *lhsFile = std::get_if<ExplicitFileDependency>(&lhs))
    return lhsFile->contentSha256.bytes() <
           std::get<ExplicitFileDependency>(rhs).contentSha256.bytes();
  const auto &lhsBundled = std::get<ToolBundledResourceDependency>(lhs);
  const auto &rhsBundled = std::get<ToolBundledResourceDependency>(rhs);
  return std::tie(lhsBundled.stableProviderBuildIdentity,
                  lhsBundled.resourceKey) <
         std::tie(rhsBundled.stableProviderBuildIdentity,
                  rhsBundled.resourceKey);
}

bool sameDependency(const ExternalDependencyIdentity &lhs,
                    const ExternalDependencyIdentity &rhs) {
  return !dependencyLess(lhs, rhs) && !dependencyLess(rhs, lhs);
}

bool externalInputLess(const ExternalInputBinding &lhs,
                       const ExternalInputBinding &rhs) {
  if (lhs.providerInputSlotRef != rhs.providerInputSlotRef)
    return lhs.providerInputSlotRef < rhs.providerInputSlotRef;
  return dependencyLess(lhs.dependencyIdentity, rhs.dependencyIdentity);
}

bool sameExternalInputs(llvm::ArrayRef<ExternalInputBinding> lhs,
                        llvm::ArrayRef<ExternalInputBinding> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  std::vector<ExternalInputBinding> canonicalLhs(lhs.begin(), lhs.end());
  std::vector<ExternalInputBinding> canonicalRhs(rhs.begin(), rhs.end());
  llvm::sort(canonicalLhs, externalInputLess);
  llvm::sort(canonicalRhs, externalInputLess);
  return llvm::all_of(
      llvm::zip_equal(canonicalLhs, canonicalRhs), [](const auto &pair) {
        return std::get<0>(pair).providerInputSlotRef ==
                   std::get<1>(pair).providerInputSlotRef &&
               sameDependency(std::get<0>(pair).dependencyIdentity,
                              std::get<1>(pair).dependencyIdentity);
      });
}

void addOccurrenceRelation(const SpecializationJob &job,
                           const ConfigurationABI &configurationAbi,
                           FabricOperationProviderOutput &output) {
  const EncodedArtifactLocalReference occurrence =
      fabric::encodeFabricArtifactLocalReference(
          ArtifactReference<fabric::FabricFuOccurrenceNodeRef>{
              configurationAbi.fabric().artifact, job.occurrence});
  for (ExternalImplementationBinding &binding :
       output.externalImplementationBindings)
    if (!llvm::is_contained(binding.fabricResourceRefs, occurrence))
      binding.fabricResourceRefs.push_back(occurrence);
}

mlir::ArrayAttr parametersOf(mlir::Operation *operation) {
  if (auto module = llvm::dyn_cast<circt::hw::HWModuleOp>(operation))
    return module.getParametersAttr();
  if (auto module = llvm::dyn_cast<circt::hw::HWModuleExternOp>(operation))
    return module.getParametersAttr();
  llvm_unreachable("concrete HW module kind was checked");
}

llvm::Error
validateProviderOutput(const SpecializationJob &job,
                       const FabricOperationProviderOutput &output) {
  const llvm::StringRef contract =
      job.provider->externalImplementationContractRef;
  if (contract.empty()) {
    if (!job.externalInputs.empty())
      return invalid("self-contained provider received external inputs");
    if (!output.externalImplementationBindings.empty())
      return invalid(
          "self-contained provider produced an external implementation "
          "binding");
    return llvm::Error::success();
  }

  if (job.externalInputs.empty())
    return invalid("external provider input closure is empty");
  if (output.externalImplementationBindings.empty())
    return invalid("external provider produced no implementation binding");
  for (const ExternalImplementationBinding &binding :
       output.externalImplementationBindings) {
    if (binding.providerContractRef != contract)
      return invalid("provider output changed the external contract ref");
    if (!sameExternalInputs(binding.externalInputs, job.externalInputs))
      return invalid("provider output changed its exact external inputs");
  }
  return llvm::Error::success();
}

bool hasBlackBoxPayload(const FabricOperationProviderOutput &output,
                        const HardwarePayloadRef &reference) {
  if (reference.role != HardwarePayloadRole::BlackBoxContract)
    return false;
  return llvm::any_of(output.payloads,
                      [&](const FabricOperationProviderPayload &payload) {
                        return payload.role == reference.role &&
                               payload.logicalName == reference.logicalName;
                      });
}

std::vector<HardwarePayload>
describePayloads(const FabricOperationProviderOutput &output) {
  std::vector<HardwarePayload> descriptors;
  descriptors.reserve(output.payloads.size());
  for (const FabricOperationProviderPayload &payload : output.payloads)
    descriptors.push_back(payload.descriptor());
  return descriptors;
}

llvm::Error
validateExternalModule(llvm::StringRef symbol,
                       const FabricOperationProviderOutput &output) {
  bool hasModuleBinding = false;
  for (const ExternalImplementationBinding &binding :
       output.externalImplementationBindings) {
    const bool locatesModule = llvm::any_of(
        binding.representationLocators,
        [&](const RepresentationLocator &locator) {
          return locator.kind == RepresentationObjectKind::Module &&
                 locator.canonicalName == symbol;
        });
    if (!locatesModule)
      continue;
    hasModuleBinding = true;
    if (binding.blackBoxContractPayloadRef &&
        hasBlackBoxPayload(output, *binding.blackBoxContractPayloadRef))
      return llvm::Error::success();
  }
  if (!hasModuleBinding)
    return invalid("external module has no exact binding");
  return invalid("external module has no exact BlackBoxContract payload");
}

llvm::Error
validatePreparedFragment(SpecializationJob &job, mlir::ModuleOp fragment,
                         const FabricOperationProviderOutput &output) {
  mlir::Operation *replacement = nullptr;
  for (mlir::Operation &operation : *fragment.getBody()) {
    if (llvm::isa<circt::hw::HWGeneratorSchemaOp>(operation))
      continue;
    if (!llvm::isa<circt::hw::HWModuleOp, circt::hw::HWModuleExternOp>(
            operation))
      return invalid("provider fragment contains a non-concrete top-level "
                     "operation");
    const auto symbol = mlir::SymbolTable::getSymbolName(&operation);
    if (!symbol)
      return invalid("provider fragment module has no symbol");
    if (llvm::isa<circt::hw::HWModuleExternOp>(operation))
      if (llvm::Error error = validateExternalModule(symbol.getValue(), output))
        return error;
    if (symbol.getValue() == job.leafSymbol) {
      if (replacement)
        return invalid("provider fragment contains duplicate concrete "
                       "replacements");
      replacement = &operation;
    }
  }
  if (!replacement)
    return invalid("provider fragment has no exact concrete replacement");

  auto replacementModule = llvm::cast<circt::hw::HWModuleLike>(replacement);
  if (replacementModule.getHWModuleType() != job.leaf.getHWModuleType())
    return invalid("provider replacement changed the leaf port contract");
  if (parametersOf(replacement) != job.leaf.getParametersAttr())
    return invalid("provider replacement changed the leaf parameters");
  if (mlir::failed(mlir::verify(fragment)))
    return invalid("provider fragment does not verify");
  return verifySpecializedCirctModule(fragment);
}

llvm::Expected<PreparedSpecialization> prepareSpecialization(
    SpecializationJob &job, const ConfigurationABI &configurationAbi,
    const platform::ImplementationPlatform *implementationPlatform) {
  mlir::OpBuilder builder(job.leaf.getContext());
  mlir::OwningOpRef<mlir::ModuleOp> fragment =
      mlir::ModuleOp::create(job.leaf.getLoc());
  builder.setInsertionPointToStart(fragment->getBody());
  mlir::Operation *schema = job.leaf.getGeneratorKindOp();
  if (!schema)
    return invalid("operation leaf generator schema is unresolved");
  builder.clone(*schema);
  auto leaf = llvm::cast<circt::hw::HWModuleGeneratedOp>(
      builder.clone(*job.leaf.getOperation()));

  auto output = job.provider->callback(FabricOperationProviderRequest{
      *fragment, leaf, job.occurrence, *job.capability, configurationAbi,
      job.recipe, implementationPlatform,
      job.provider->externalImplementationContractRef, job.externalInputs});
  if (!output)
    return output.takeError();
  addOccurrenceRelation(job, configurationAbi, *output);
  if (llvm::Error error = validateProviderOutput(job, *output))
    return std::move(error);
  if (llvm::Error error = validatePreparedFragment(job, *fragment, *output))
    return std::move(error);
  return PreparedSpecialization{&job, std::move(fragment), std::move(*output)};
}

llvm::Error
validateSymbolClosure(mlir::ModuleOp module,
                      llvm::ArrayRef<PreparedSpecialization> specializations) {
  std::map<std::string, std::string> materializedSymbols;
  for (const PreparedSpecialization &specialization : specializations) {
    mlir::ModuleOp fragment = specialization.fragment.get();
    for (mlir::Operation &operation : *fragment.getBody()) {
      if (llvm::isa<circt::hw::HWGeneratorSchemaOp>(operation))
        continue;
      const std::string symbol =
          mlir::SymbolTable::getSymbolName(&operation).getValue().str();
      if (!materializedSymbols.emplace(symbol, specialization.job->leafSymbol)
               .second)
        return invalid("provider fragments contain a duplicate symbol");
      mlir::Operation *existing = module.lookupSymbol(symbol);
      if (existing && existing != specialization.job->leaf.getOperation())
        return invalid("provider fragment symbol collides with the common "
                       "skeleton");
    }
  }
  return llvm::Error::success();
}

llvm::Error
applySpecializations(mlir::ModuleOp module,
                     llvm::ArrayRef<PreparedSpecialization> specializations) {
  for (const PreparedSpecialization &specialization : specializations) {
    auto leaf = module.lookupSymbol<circt::hw::HWModuleGeneratedOp>(
        specialization.job->leafSymbol);
    if (!leaf)
      return invalid("prepared operation leaf is absent from working module");
    mlir::OpBuilder builder(leaf.getContext());
    builder.setInsertionPoint(leaf);
    mlir::ModuleOp fragment = specialization.fragment.get();
    for (mlir::Operation &operation : *fragment.getBody()) {
      if (!llvm::isa<circt::hw::HWGeneratorSchemaOp>(operation))
        builder.clone(operation);
    }
    leaf.erase();
  }
  return llvm::Error::success();
}

void appendOutput(FabricOperationProviderOutput &destination,
                  FabricOperationProviderOutput source) {
  destination.payloads.insert(destination.payloads.end(),
                              std::make_move_iterator(source.payloads.begin()),
                              std::make_move_iterator(source.payloads.end()));
  destination.activityPoints.insert(
      destination.activityPoints.end(),
      std::make_move_iterator(source.activityPoints.begin()),
      std::make_move_iterator(source.activityPoints.end()));
  destination.externalImplementationBindings.insert(
      destination.externalImplementationBindings.end(),
      std::make_move_iterator(source.externalImplementationBindings.begin()),
      std::make_move_iterator(source.externalImplementationBindings.end()));
}

} // namespace

HardwarePayload FabricOperationProviderPayload::descriptor() const {
  return HardwarePayload{role, logicalName, mediaType,
                         computeBlobDigest(bytes)};
}

llvm::StringRef backendRecipeKeyword(BackendRecipeKey recipe) {
  switch (recipe) {
  case BackendRecipeKey::PortableSystemVerilog:
    return "portable_system_verilog";
  case BackendRecipeKey::SynopsysDesignWare:
    return "synopsys_designware";
  case BackendRecipeKey::CadenceChipWare:
    return "cadence_chipware";
  case BackendRecipeKey::AmdXilinx:
    return "amd_xilinx";
  case BackendRecipeKey::IntelAltera:
    return "intel_altera";
  }
  llvm_unreachable("invalid backend recipe key");
}

llvm::Error FabricOperationProviderRegistry::add(
    FabricOperationProviderRegistration registration) {
  if (!validFamily(registration.implementationFamily))
    return invalid("provider registration has an unknown implementation "
                   "family");
  if (!validRecipe(registration.recipe))
    return invalid("provider registration has an unknown backend recipe");
  if (!registration.callback)
    return invalid("provider registration has no callback");

  auto insertion = llvm::lower_bound(
      registrations_, registrationKey(registration),
      [](const FabricOperationProviderRegistration &entry, const auto &key) {
        return registrationKey(entry) < key;
      });
  if (insertion != registrations_.end() &&
      registrationKey(*insertion) == registrationKey(registration))
    return invalid("provider registration key is a duplicate");
  registrations_.insert(insertion, registration);
  return llvm::Error::success();
}

const FabricOperationProviderRegistration *
FabricOperationProviderRegistry::find(
    ::fabric::ImplementationFamilyId implementationFamily,
    BackendRecipeKey recipe) const {
  FabricOperationProviderRegistration requested{
      implementationFamily, recipe, {}, nullptr};
  auto found = llvm::lower_bound(
      registrations_, registrationKey(requested),
      [](const FabricOperationProviderRegistration &entry, const auto &key) {
        return registrationKey(entry) < key;
      });
  if (found == registrations_.end() ||
      registrationKey(*found) != registrationKey(requested))
    return nullptr;
  return &*found;
}

std::vector<FabricOperationProviderCoverage>
FabricOperationProviderRegistry::coverage() const {
  std::vector<FabricOperationProviderCoverage> result;
  result.reserve(::fabric::implementationFamilyCount());
  auto registration = registrations_.begin();
  for (std::uint32_t ordinal = 0;
       ordinal < ::fabric::implementationFamilyCount(); ++ordinal) {
    auto family = static_cast<::fabric::ImplementationFamilyId>(ordinal);
    FabricOperationProviderCoverage entry{family, {}};
    while (registration != registrations_.end() &&
           registration->implementationFamily == family) {
      entry.recipes.push_back(registration->recipe);
      ++registration;
    }
    result.push_back(std::move(entry));
  }
  return result;
}

char FabricOperationProviderUnsupportedError::ID = 0;

void FabricOperationProviderUnsupportedError::log(
    llvm::raw_ostream &stream) const {
  stream << "rtl_operation_provider_unsupported: family '"
         << ::fabric::implementationFamilyKeyword(implementationFamily_)
         << "' has no '" << backendRecipeKeyword(recipe_) << "' provider";
}

std::error_code
FabricOperationProviderUnsupportedError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Expected<FabricOperationProviderOutput> specializeFabricOperationLeaves(
    mlir::ModuleOp module, const fabric::FinalizedFabricRoot &fabric,
    const FinalizedConfigurationABI &configurationAbi,
    llvm::ArrayRef<FabricOperationLeafAssociation> operationLeaves,
    llvm::ArrayRef<FabricOperationRecipeBinding> operationRecipes,
    const FabricOperationProviderRegistry &providers,
    const ExternalImplementationContractCatalog &externalContracts,
    const platform::ImplementationPlatform *implementationPlatform) {
  if (configurationAbi.abi().fabric() != fabric.reference())
    return invalid("ConfigurationABI does not implement the exact Fabric");
  if (llvm::Error error =
          verifyCommonCirctSkeleton(module, fabric.view(), operationLeaves))
    return error;

  std::map<std::vector<std::uint8_t>, const FabricOperationRecipeBinding *>
      recipes;
  for (const FabricOperationRecipeBinding &binding : operationRecipes) {
    if (!validRecipe(binding.recipe))
      return invalid("occurrence recipe binding has an unknown recipe");
    if (llvm::Error error =
            fabric::validateFabricRef(fabric.view(), binding.occurrence)) {
      llvm::consumeError(std::move(error));
      return invalid("occurrence recipe binding does not resolve in Fabric");
    }
    auto inserted = recipes.emplace(
        fabric::canonicalFabricBytes(binding.occurrence), &binding);
    if (!inserted.second)
      return invalid("occurrence recipe binding is a duplicate");
  }
  if (recipes.size() != operationLeaves.size())
    return invalid("recipe bindings do not cover every operation leaf");

  std::vector<SpecializationJob> jobs;
  jobs.reserve(operationLeaves.size());
  for (const FabricOperationLeafAssociation &association : operationLeaves) {
    circt::hw::HWModuleGeneratedOp leaf = association.module;
    std::vector<std::uint8_t> occurrenceKey =
        fabric::canonicalFabricBytes(association.occurrence);
    auto recipe = recipes.find(occurrenceKey);
    if (recipe == recipes.end())
      return invalid("recipe bindings do not match the operation leaves");
    const auto *capability =
        fabric.view().resolvedFabricOpCapability(association.occurrence);
    if (!capability)
      return invalid("operation leaf has no resolved Fabric capability");
    const FabricOperationProviderRegistration *provider = providers.find(
        capability->implementationFamily, recipe->second->recipe);
    if (!provider)
      return llvm::make_error<FabricOperationProviderUnsupportedError>(
          capability->implementationFamily, recipe->second->recipe);
    std::vector<ExternalInputBinding> externalInputs;
    if (provider->externalImplementationContractRef.empty()) {
      if (!recipe->second->externalInputs.empty())
        return invalid("self-contained provider received external inputs");
    } else {
      auto canonicalInputs = externalContracts.canonicalizeAndValidateInputs(
          provider->externalImplementationContractRef,
          recipe->second->externalInputs, HardwareRepresentation::Rtl);
      if (!canonicalInputs)
        return canonicalInputs.takeError();
      externalInputs = std::move(*canonicalInputs);
    }
    jobs.push_back(SpecializationJob{
        std::move(occurrenceKey), leaf, leaf.getSymName().str(),
        association.occurrence, capability, provider, std::move(externalInputs),
        recipe->second->recipe});
  }

  llvm::sort(jobs,
             [](const SpecializationJob &lhs, const SpecializationJob &rhs) {
               return lhs.occurrenceKey < rhs.occurrenceKey;
             });
  std::vector<PreparedSpecialization> prepared;
  prepared.reserve(jobs.size());
  for (SpecializationJob &job : jobs) {
    auto specialization = prepareSpecialization(job, configurationAbi.abi(),
                                                implementationPlatform);
    if (!specialization)
      return specialization.takeError();
    prepared.push_back(std::move(*specialization));
  }
  if (llvm::Error error = validateSymbolClosure(module, prepared))
    return std::move(error);

  FabricOperationProviderOutput output;
  for (PreparedSpecialization &specialization : prepared)
    appendOutput(output, std::move(specialization.output));
  const std::vector<HardwarePayload> payloads = describePayloads(output);
  if (llvm::Error error = externalContracts.canonicalizeAndValidateBindings(
          output.externalImplementationBindings, HardwareRepresentation::Rtl,
          implementationPlatform, payloads, fabric.view()))
    return std::move(error);

  mlir::OwningOpRef<mlir::ModuleOp> working(
      llvm::cast<mlir::ModuleOp>(module->clone()));
  if (llvm::Error error = applySpecializations(*working, prepared))
    return std::move(error);
  if (llvm::Error error = verifySpecializedCirctModule(*working))
    return std::move(error);

  module->setAttrs((*working)->getAttrs());
  module.getBodyRegion().takeBody(working->getBodyRegion());
  return output;
}

} // namespace loom::hardware::rtl
