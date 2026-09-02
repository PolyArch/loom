#include "Hardware/RTL/Specialization.h"

#include "Fabric/IR/ResourceContractRecord.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Hardware/RTL/MaterializationDiagnostics.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
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
  circt::hw::HWModuleGeneratedOp leaf;
  std::string leafSymbol;
  fabric::FabricPhysicalOccurrenceOwnerRef occurrence;
  const fabric::ResolvedFabricOpCapabilityView *capability = nullptr;
  const FabricOperationProviderRegistration *provider = nullptr;
  std::vector<ExternalInputBinding> externalInputs;
  BackendRecipeKey recipe = BackendRecipeKey::PortableSystemVerilog;
};

struct OrderedOperationLeaf final {
  std::vector<std::uint8_t> occurrenceKey;
  const FabricOperationLeafAssociation *association = nullptr;
};

struct PreparedSpecialization final {
  SpecializationJob *job = nullptr;
  mlir::OwningOpRef<mlir::ModuleOp> fragment;
  FabricOperationProviderOutput output;
};

struct RetargetedInstance final {
  circt::hw::InstanceOp instance;
  mlir::StringAttr moduleName;
  mlir::ArrayAttr argumentNames;
  mlir::ArrayAttr resultNames;
};

struct AppliedSpecialization final {
  mlir::OwningOpRef<mlir::Operation *> originalLeaf;
  mlir::Block *originalBlock = nullptr;
  mlir::Operation *originalSuccessor = nullptr;
  std::vector<mlir::Operation *> replacements;
  std::vector<RetargetedInstance> retargetedInstances;
};

struct SharedProviderImplementation final {
  std::string symbol;
  fabric::FabricPhysicalOccurrenceOwnerRef canonicalOccurrence;
  std::vector<ActivityPoint> activityPoints;
  std::vector<ExternalImplementationBindingDraft>
      externalImplementationBindings;
};

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendBytes(std::vector<std::uint8_t> &bytes,
                 llvm::ArrayRef<std::uint8_t> value) {
  appendU64(bytes, value.size());
  bytes.insert(bytes.end(), value.begin(), value.end());
}

void appendString(std::vector<std::uint8_t> &bytes, llvm::StringRef value) {
  appendBytes(bytes, llvm::ArrayRef<std::uint8_t>(
                         reinterpret_cast<const std::uint8_t *>(value.data()),
                         value.size()));
}

std::string printed(mlir::Attribute attribute) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  attribute.print(stream);
  return result;
}

std::string printed(mlir::Type type) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  type.print(stream);
  return result;
}

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

void appendExternalInputKey(std::vector<std::uint8_t> &key,
                            const ExternalInputBinding &input) {
  appendString(key, input.providerInputSlotRef);
  appendU32(key, static_cast<std::uint32_t>(input.dependencyIdentity.index()));
  if (const auto *file =
          std::get_if<ExplicitFileDependency>(&input.dependencyIdentity)) {
    const auto digest = file->contentSha256.bytes();
    appendBytes(key,
                llvm::ArrayRef<std::uint8_t>(digest.data(), digest.size()));
    return;
  }
  const auto &bundled =
      std::get<ToolBundledResourceDependency>(input.dependencyIdentity);
  appendString(key, bundled.stableProviderBuildIdentity);
  appendString(key, bundled.resourceKey);
}

void appendEncodingRelationKey(std::vector<std::uint8_t> &key,
                               const ConfigurationEncodingRelation &relation) {
  appendU32(key, static_cast<std::uint32_t>(relation.semanticEncoding.index()));
  std::visit(
      [&](const auto &encoding) {
        appendU64(key, encoding.encodedBitCount);
        using Encoding = std::decay_t<decltype(encoding)>;
        if constexpr (std::is_same_v<Encoding, FiniteCodebookEncoding>) {
          appendU64(key, encoding.entries.size());
          for (const FiniteCodebookEntry &entry : encoding.entries) {
            appendBytes(key, entry.semanticValue);
            appendBytes(key, entry.physicalCode);
          }
        }
      },
      relation.semanticEncoding);
  appendBytes(key, relation.inactiveValue);
}

void appendImplementationPlatformKey(
    std::vector<std::uint8_t> &key,
    const platform::ImplementationPlatform *implementationPlatform) {
  appendU32(key, implementationPlatform ? 1 : 0);
  if (!implementationPlatform)
    return;

  const platform::ImplementationTarget &target =
      implementationPlatform->target();
  appendU32(key, static_cast<std::uint32_t>(target.index()));
  std::visit(
      [&](const auto &value) {
        using Target = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Target, platform::AsicTarget>) {
          appendString(key, value.technologyIdentity);
          appendString(key, value.releaseIdentity);
        } else {
          appendU32(key, static_cast<std::uint32_t>(value.vendor));
          appendString(key, value.deviceOrderingCode);
        }
      },
      target);

  appendU64(key, implementationPlatform->technologyCorners().size());
  for (const platform::TechnologyCorner &corner :
       implementationPlatform->technologyCorners()) {
    appendU64(key, corner.id.value());
    appendString(key, corner.key);
  }
}

llvm::Expected<std::vector<std::uint8_t>> implementationKey(
    SpecializationJob &job, const ConfigurationABI &configurationAbi,
    const platform::ImplementationPlatform *implementationPlatform) {
  std::vector<std::uint8_t> key;
  const fabric::ResolvedFabricOpCapabilityView &capability = *job.capability;

  // Registration identity is the registry's closed, unique family/recipe key.
  appendU32(key,
            static_cast<std::uint32_t>(job.provider->implementationFamily));
  appendU32(key, static_cast<std::uint32_t>(job.provider->recipe));
  appendString(key, job.provider->externalImplementationContractRef);
  appendU32(key, static_cast<std::uint32_t>(job.recipe));

  // Project owner-qualified references out of the immutable capability view.
  // Occurrence ownership is accumulated separately in provider output.
  appendU32(key, static_cast<std::uint32_t>(capability.implementationFamily));
  appendString(key,
               printed(::fabric::getFamilyCapabilityParamsAttr(
                   job.leaf.getContext(), capability.parameterizedCapability)));
  appendU64(key, capability.enabledOperationSchemas.size());
  for (::dataflow::OperationSchemaId schema :
       capability.enabledOperationSchemas)
    appendU32(key, static_cast<std::uint32_t>(schema));

  appendU64(key, capability.physicalPorts.size());
  for (const fabric::ResolvedFabricOpPhysicalPortView &port :
       capability.physicalPorts) {
    appendU32(key, static_cast<std::uint32_t>(port.reference.direction));
    appendU64(key, port.reference.ordinal);
    appendU32(key, port.payloadWidthBits);
    appendBytes(key, port.canonicalType);
  }

  appendU64(key, capability.configurationFieldSchema.size());
  for (const fabric::FabricSemanticConfigFieldRef &field :
       capability.configurationFieldSchema) {
    appendU64(key, field.ordinal);
    const ConfigurationEncodingRelation *relation =
        configurationAbi.findOperationEncodingRelation(job.occurrence,
                                                       field.ordinal);
    if (!relation)
      return invalid("operation configuration relation is unresolved");
    appendEncodingRelationKey(key, *relation);
  }

  auto resource = ::fabric::encodeResourceContractRecord(
      capability.resourceStateAndTimingContract);
  if (!resource)
    return resource.takeError();
  appendBytes(key, *resource);
  appendU64(key, capability.physicalRefinementDomains.size());
  for (const fabric::FabricPhysicalRefinementDomainRef &domain :
       capability.physicalRefinementDomains)
    appendU64(key, domain.ordinal);

  appendString(key, printed(job.leaf.getHWModuleType()));
  appendString(key, printed(job.leaf.getParametersAttr()));

  std::vector<ExternalInputBinding> inputs(job.externalInputs.begin(),
                                           job.externalInputs.end());
  llvm::sort(inputs, externalInputLess);
  appendU64(key, inputs.size());
  for (const ExternalInputBinding &input : inputs)
    appendExternalInputKey(key, input);
  appendImplementationPlatformKey(key, implementationPlatform);
  return key;
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
                           FabricOperationProviderOutput &output) {
  for (ExternalImplementationBindingDraft &binding :
       output.externalImplementationBindings)
    if (!llvm::is_contained(binding.fabricResourceRefs, job.occurrence))
      binding.fabricResourceRefs.push_back(job.occurrence);
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
  for (const ActivityPoint &point : output.activityPoints)
    if (point.semanticFabricRef && *point.semanticFabricRef != job.occurrence)
      return invalid("provider activity point has foreign occurrence "
                     "ownership");

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
  for (const ExternalImplementationBindingDraft &binding :
       output.externalImplementationBindings) {
    if (!binding.fabricResourceRefs.empty())
      return invalid("provider output introduced occurrence ownership");
    if (binding.providerContractRef != contract)
      return invalid("provider output changed the external contract ref");
    if (!sameExternalInputs(binding.externalInputs, job.externalInputs))
      return invalid("provider output changed its exact external inputs");
  }
  return llvm::Error::success();
}

bool hasBlackBoxPayload(const FabricOperationProviderOutput &output,
                        const ImplementationPayloadKey &reference) {
  if (reference.role != PayloadRole::BlackBoxContract)
    return false;
  return llvm::any_of(
      output.payloads, [&](const FabricOperationProviderPayload &payload) {
        return payload.role == reference.role &&
               payload.canonicalLogicalName == reference.canonicalLogicalName;
      });
}

llvm::Error
validateExternalModule(llvm::StringRef symbol,
                       const FabricOperationProviderOutput &output) {
  bool hasModuleBinding = false;
  for (const ExternalImplementationBindingDraft &binding :
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
    if (binding.blackBoxContractPayload &&
        hasBlackBoxPayload(output, *binding.blackBoxContractPayload))
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
  if (llvm::Error error = validateProviderOutput(job, *output))
    return std::move(error);
  addOccurrenceRelation(job, *output);
  if (llvm::Error error = validatePreparedFragment(job, *fragment, *output))
    return std::move(error);
  return PreparedSpecialization{&job, std::move(fragment), std::move(*output)};
}

llvm::Error
validateSymbolClosure(mlir::ModuleOp module,
                      const PreparedSpecialization &specialization) {
  std::map<std::string, std::string> materializedSymbols;
  mlir::ModuleOp fragment = specialization.fragment.get();
  for (mlir::Operation &operation : *fragment.getBody()) {
    if (llvm::isa<circt::hw::HWGeneratorSchemaOp>(operation))
      continue;
    const std::string symbol =
        mlir::SymbolTable::getSymbolName(&operation).getValue().str();
    if (!materializedSymbols.emplace(symbol, specialization.job->leafSymbol)
             .second)
      return invalid("provider fragment contains a duplicate symbol");
    mlir::Operation *existing = module.lookupSymbol(symbol);
    if (existing && existing != specialization.job->leaf.getOperation())
      return invalid("provider fragment symbol collides with the common "
                     "skeleton");
  }
  return llvm::Error::success();
}

void rollbackSpecializations(
    llvm::MutableArrayRef<AppliedSpecialization> specializations) noexcept {
  for (AppliedSpecialization &specialization : llvm::reverse(specializations)) {
    assert(specialization.originalLeaf && specialization.originalBlock &&
           "specialization rollback lost its original leaf");
    for (RetargetedInstance &retarget : specialization.retargetedInstances) {
      retarget.instance.setModuleName(retarget.moduleName);
      retarget.instance.setArgNamesAttr(retarget.argumentNames);
      retarget.instance.setResultNamesAttr(retarget.resultNames);
    }
    mlir::Operation *leaf = specialization.originalLeaf.release();
    if (!specialization.replacements.empty()) {
      mlir::Operation *firstReplacement = specialization.replacements.front();
      firstReplacement->getBlock()->getOperations().insert(
          firstReplacement->getIterator(), leaf);
    } else if (specialization.originalSuccessor &&
               specialization.originalSuccessor->getBlock() ==
                   specialization.originalBlock) {
      specialization.originalBlock->getOperations().insert(
          specialization.originalSuccessor->getIterator(), leaf);
    } else {
      specialization.originalBlock->getOperations().push_back(leaf);
    }
    for (mlir::Operation *replacement :
         llvm::reverse(specialization.replacements))
      replacement->erase();
  }
}

llvm::Expected<AppliedSpecialization>
applySpecialization(mlir::ModuleOp module,
                    PreparedSpecialization &specialization) {
  auto leaf = module.lookupSymbol<circt::hw::HWModuleGeneratedOp>(
      specialization.job->leafSymbol);
  if (!leaf)
    return invalid("prepared operation leaf is absent from working module");
  AppliedSpecialization change;
  mlir::ModuleOp fragment = specialization.fragment.get();
  const bool hasConcreteModule =
      llvm::any_of(*fragment.getBody(), [](mlir::Operation &operation) {
        return !llvm::isa<circt::hw::HWGeneratorSchemaOp>(operation);
      });
  if (!hasConcreteModule)
    return invalid("prepared provider fragment has no concrete module");
  mlir::Block *block = leaf->getBlock();
  mlir::Operation *successor = leaf->getNextNode();
  change.originalBlock = block;
  change.originalSuccessor = successor;
  leaf->remove();
  change.originalLeaf =
      mlir::OwningOpRef<mlir::Operation *>(leaf.getOperation());
  mlir::OpBuilder builder(leaf.getContext());
  if (successor)
    builder.setInsertionPoint(successor);
  else
    builder.setInsertionPointToEnd(block);
  for (mlir::Operation &operation : *fragment.getBody()) {
    if (!llvm::isa<circt::hw::HWGeneratorSchemaOp>(operation))
      change.replacements.push_back(builder.clone(operation));
  }
  specialization.fragment = nullptr;
  return change;
}

llvm::Expected<AppliedSpecialization> applySharedImplementationReference(
    mlir::ModuleOp module, const SpecializationJob &job,
    llvm::StringRef sharedSymbol,
    llvm::ArrayRef<circt::hw::InstanceOp> instances) {
  auto leaf =
      module.lookupSymbol<circt::hw::HWModuleGeneratedOp>(job.leafSymbol);
  if (!leaf)
    return invalid("reused operation leaf is absent from working module");
  mlir::Operation *sharedOperation = module.lookupSymbol(sharedSymbol);
  auto shared =
      llvm::dyn_cast_or_null<circt::hw::HWModuleLike>(sharedOperation);
  if (!shared)
    return invalid("shared provider implementation is absent");
  if (shared.getHWModuleType() != leaf.getHWModuleType() ||
      parametersOf(sharedOperation) != leaf.getParametersAttr())
    return invalid("shared provider implementation changed its module "
                   "contract");

  AppliedSpecialization change;
  change.originalBlock = leaf->getBlock();
  change.originalSuccessor = leaf->getNextNode();
  for (circt::hw::InstanceOp instance : instances) {
    change.retargetedInstances.push_back({instance, leaf.getSymNameAttr(),
                                          instance.getArgNamesAttr(),
                                          instance.getResultNamesAttr()});
    instance.setModuleName(shared.getModuleNameAttr());
  }
  leaf->remove();
  change.originalLeaf =
      mlir::OwningOpRef<mlir::Operation *>(leaf.getOperation());
  return change;
}

std::string normalizedInternalModuleKey(circt::hw::HWModuleOp shell) {
  mlir::OwningOpRef<mlir::Operation *> clone(shell->clone());
  auto normalized = llvm::cast<circt::hw::HWModuleOp>(clone.get());
  normalized->setAttr(
      mlir::SymbolTable::getSymbolAttrName(),
      mlir::StringAttr::get(shell.getContext(), "operation_shell"));
  llvm::SmallVector<mlir::Attribute> portNames;
  portNames.reserve(normalized.getNumPorts());
  for (std::size_t port = 0; port != normalized.getNumPorts(); ++port)
    portNames.push_back(mlir::StringAttr::get(shell.getContext(),
                                              "port_" + std::to_string(port)));
  normalized.setAllPortNames(portNames);
  std::string key;
  llvm::raw_string_ostream stream(key);
  mlir::Type(normalized.getHWModuleType()).print(stream);
  stream << '\n';
  mlir::OpPrintingFlags flags;
  flags.printGenericOpForm();
  normalized.print(stream, flags);
  stream.flush();
  return key;
}

llvm::Expected<AppliedSpecialization> applySharedInternalModuleReference(
    circt::hw::HWModuleOp shell, circt::hw::HWModuleOp shared,
    llvm::ArrayRef<circt::hw::InstanceOp> instances) {
  const auto shellPorts = shell.getPortList();
  const auto sharedPortList = shared.getPortList();
  if (shellPorts.size() != sharedPortList.size() ||
      shell.getParametersAttr() != shared.getParametersAttr() ||
      !llvm::all_of(
          llvm::zip_equal(shellPorts, sharedPortList), [](const auto &ports) {
            const auto &[lhs, rhs] = ports;
            return lhs.isOutput() == rhs.isOutput() && lhs.type == rhs.type;
          }))
    return invalid("shared internal module changed its contract");

  circt::hw::ModulePortInfo sharedPorts(shared.getPortList());
  llvm::SmallVector<mlir::Attribute> argumentNames;
  llvm::SmallVector<mlir::Attribute> resultNames;
  for (const circt::hw::PortInfo &port : sharedPorts.getInputs())
    argumentNames.push_back(
        mlir::StringAttr::get(shell.getContext(), port.getName()));
  for (const circt::hw::PortInfo &port : sharedPorts.getOutputs())
    resultNames.push_back(
        mlir::StringAttr::get(shell.getContext(), port.getName()));
  const mlir::ArrayAttr sharedArgumentNames =
      mlir::ArrayAttr::get(shell.getContext(), argumentNames);
  const mlir::ArrayAttr sharedResultNames =
      mlir::ArrayAttr::get(shell.getContext(), resultNames);

  AppliedSpecialization change;
  change.originalBlock = shell->getBlock();
  change.originalSuccessor = shell->getNextNode();
  for (circt::hw::InstanceOp instance : instances) {
    change.retargetedInstances.push_back({instance, shell.getSymNameAttr(),
                                          instance.getArgNamesAttr(),
                                          instance.getResultNamesAttr()});
    instance.setModuleName(shared.getModuleNameAttr());
    instance.setArgNamesAttr(sharedArgumentNames);
    instance.setResultNamesAttr(sharedResultNames);
  }
  shell->remove();
  change.originalLeaf =
      mlir::OwningOpRef<mlir::Operation *>(shell.getOperation());
  return change;
}

llvm::Error internInternalModules(
    llvm::ArrayRef<circt::hw::HWModuleOp> shells,
    const std::map<std::string, std::vector<circt::hw::InstanceOp>>
        &instancesByModule,
    std::vector<AppliedSpecialization> &applied) {
  std::map<std::string, circt::hw::HWModuleOp> definitions;
  for (circt::hw::HWModuleOp shell : shells) {
    const std::string key = normalizedInternalModuleKey(shell);
    auto [definition, inserted] = definitions.emplace(key, shell);
    if (inserted)
      continue;
    const auto instances = instancesByModule.find(shell.getSymName().str());
    auto change = applySharedInternalModuleReference(
        shell, definition->second,
        instances == instancesByModule.end()
            ? llvm::ArrayRef<circt::hw::InstanceOp>{}
            : llvm::ArrayRef(instances->second));
    if (!change)
      return change.takeError();
    applied.push_back(std::move(*change));
  }
  return llvm::Error::success();
}

std::vector<circt::hw::HWModuleOp>
parentModulesOf(llvm::ArrayRef<circt::hw::HWModuleOp> modules,
                const std::map<std::string, std::vector<circt::hw::InstanceOp>>
                    &instancesByModule) {
  std::vector<circt::hw::HWModuleOp> result;
  std::set<mlir::Operation *> seen;
  for (circt::hw::HWModuleOp module : modules) {
    const auto instances = instancesByModule.find(module.getSymName().str());
    if (instances == instancesByModule.end())
      continue;
    for (circt::hw::InstanceOp instance : instances->second) {
      auto parent = instance->getParentOfType<circt::hw::HWModuleOp>();
      if (parent && seen.insert(parent.getOperation()).second)
        result.push_back(parent);
    }
  }
  llvm::sort(result, [](circt::hw::HWModuleOp lhs, circt::hw::HWModuleOp rhs) {
    return lhs.getSymName() < rhs.getSymName();
  });
  return result;
}

FabricOperationProviderOutput occurrenceRelationsForReuse(
    const SharedProviderImplementation &shared,
    const fabric::FabricPhysicalOccurrenceOwnerRef &occurrence) {
  FabricOperationProviderOutput output;
  for (const ActivityPoint &point : shared.activityPoints) {
    if (!point.semanticFabricRef ||
        *point.semanticFabricRef != shared.canonicalOccurrence)
      continue;
    ActivityPoint projected = point;
    projected.semanticFabricRef = occurrence;
    output.activityPoints.push_back(std::move(projected));
  }
  output.externalImplementationBindings = shared.externalImplementationBindings;
  for (ExternalImplementationBindingDraft &binding :
       output.externalImplementationBindings)
    binding.fabricResourceRefs = {occurrence};
  return output;
}

bool sameLocatorSet(llvm::ArrayRef<RepresentationLocator> lhs,
                    llvm::ArrayRef<RepresentationLocator> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  std::vector<RepresentationLocator> canonicalLhs(lhs.begin(), lhs.end());
  std::vector<RepresentationLocator> canonicalRhs(rhs.begin(), rhs.end());
  const auto less = [](const RepresentationLocator &a,
                       const RepresentationLocator &b) {
    return std::tie(a.kind, a.canonicalName) <
           std::tie(b.kind, b.canonicalName);
  };
  llvm::sort(canonicalLhs, less);
  llvm::sort(canonicalRhs, less);
  return canonicalLhs == canonicalRhs;
}

bool sameExternalDefinition(const ExternalImplementationBindingDraft &lhs,
                            const ExternalImplementationBindingDraft &rhs) {
  return lhs.providerContractRef == rhs.providerContractRef &&
         sameExternalInputs(lhs.externalInputs, rhs.externalInputs) &&
         sameLocatorSet(lhs.representationLocators,
                        rhs.representationLocators) &&
         lhs.blackBoxContractPayload == rhs.blackBoxContractPayload;
}

void canonicalizePhysicalOwners(
    std::vector<fabric::FabricPhysicalOccurrenceOwnerRef> &owners) {
  llvm::sort(owners, [](const auto &lhs, const auto &rhs) {
    return fabric::canonicalFabricBytes(lhs) <
           fabric::canonicalFabricBytes(rhs);
  });
  owners.erase(std::unique(owners.begin(), owners.end()), owners.end());
}

llvm::Error appendOutput(FabricOperationProviderOutput &destination,
                         FabricOperationProviderOutput source) {
  for (FabricOperationProviderPayload &payload : source.payloads) {
    auto existing = llvm::find_if(
        destination.payloads,
        [&](const FabricOperationProviderPayload &candidate) {
          return candidate.role == payload.role &&
                 candidate.canonicalLogicalName == payload.canonicalLogicalName;
        });
    if (existing == destination.payloads.end()) {
      destination.payloads.push_back(std::move(payload));
      continue;
    }
    if (existing->bytes != payload.bytes)
      return invalid("provider payload key has conflicting bytes");
  }
  destination.activityPoints.insert(
      destination.activityPoints.end(),
      std::make_move_iterator(source.activityPoints.begin()),
      std::make_move_iterator(source.activityPoints.end()));
  for (ExternalImplementationBindingDraft &binding :
       source.externalImplementationBindings) {
    auto existing =
        llvm::find_if(destination.externalImplementationBindings,
                      [&](const ExternalImplementationBindingDraft &candidate) {
                        return sameExternalDefinition(candidate, binding);
                      });
    if (existing == destination.externalImplementationBindings.end()) {
      canonicalizePhysicalOwners(binding.fabricResourceRefs);
      destination.externalImplementationBindings.push_back(std::move(binding));
      continue;
    }
    existing->fabricResourceRefs.insert(existing->fabricResourceRefs.end(),
                                        binding.fabricResourceRefs.begin(),
                                        binding.fabricResourceRefs.end());
    canonicalizePhysicalOwners(existing->fabricResourceRefs);
  }
  return llvm::Error::success();
}

} // namespace

ImplementationPayload FabricOperationProviderPayload::descriptor() const {
  return ImplementationPayload{role, canonicalLogicalName,
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
         << "' cannot implement the exact capability with recipe '"
         << backendRecipeKeyword(recipe_) << "'";
}

std::error_code
FabricOperationProviderUnsupportedError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Expected<FabricOperationProviderOutput> specializeFabricOperationLeaves(
    mlir::ModuleOp module, const FinalizedConfigurationABI &configurationAbi,
    llvm::ArrayRef<FabricOperationLeafAssociation> operationLeaves,
    llvm::ArrayRef<FabricOperationRecipeBinding> operationRecipes,
    const FabricOperationProviderRegistry &providers,
    const ExternalImplementationContractCatalog &externalContracts,
    const platform::ImplementationPlatform *implementationPlatform,
    llvm::StringRef materializationKey) {
  {
    RtlMaterializationStageTracker inputVerify("specialization_input_verify",
                                               materializationKey, module);
    if (llvm::Error error = verifyCommonCirctSkeleton(
            module, configurationAbi.abi(), operationLeaves))
      return error;
    inputVerify.finish(module);
  }

  RtlMaterializationStageTracker preflight(
      "specialization_preflight_job_closure", materializationKey, module);

  std::map<std::vector<std::uint8_t>, const FabricOperationRecipeBinding *>
      recipes;
  for (const FabricOperationRecipeBinding &binding : operationRecipes) {
    if (!validRecipe(binding.recipe))
      return invalid("occurrence recipe binding has an unknown recipe");
    auto operation = resolveFabricPhysicalOperation(
        configurationAbi.abi().fabricSystem(), binding.occurrence);
    if (!operation) {
      llvm::consumeError(operation.takeError());
      return invalid("occurrence recipe binding does not resolve in Fabric");
    }
    auto inserted = recipes.emplace(
        fabric::canonicalFabricBytes(binding.occurrence), &binding);
    if (!inserted.second)
      return invalid("occurrence recipe binding is a duplicate");
  }
  if (recipes.size() != operationLeaves.size())
    return invalid("recipe bindings do not cover every operation leaf");

  std::vector<OrderedOperationLeaf> orderedLeaves;
  orderedLeaves.reserve(operationLeaves.size());
  for (const FabricOperationLeafAssociation &association : operationLeaves) {
    orderedLeaves.push_back(OrderedOperationLeaf{
        fabric::canonicalFabricBytes(association.occurrence), &association});
  }
  llvm::sort(orderedLeaves, [](const OrderedOperationLeaf &lhs,
                               const OrderedOperationLeaf &rhs) {
    return lhs.occurrenceKey < rhs.occurrenceKey;
  });

  const auto resolveJob = [&](const OrderedOperationLeaf &ordered)
      -> llvm::Expected<SpecializationJob> {
    const FabricOperationLeafAssociation &association = *ordered.association;
    circt::hw::HWModuleGeneratedOp leaf = association.module;
    auto recipe = recipes.find(ordered.occurrenceKey);
    if (recipe == recipes.end())
      return invalid("recipe bindings do not match the operation leaves");
    auto operation = resolveFabricPhysicalOperation(
        configurationAbi.abi().fabricSystem(), association.occurrence);
    if (!operation)
      return invalid("operation leaf has no resolved Fabric capability");
    const auto *capability = operation->capability;
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
          recipe->second->externalInputs, RepresentationRootVariant::Rtl);
      if (!canonicalInputs)
        return canonicalInputs.takeError();
      externalInputs = std::move(*canonicalInputs);
    }
    return SpecializationJob{
        leaf,     leaf.getSymName().str(),   association.occurrence, capability,
        provider, std::move(externalInputs), recipe->second->recipe};
  };

  for (const OrderedOperationLeaf &ordered : orderedLeaves) {
    auto job = resolveJob(ordered);
    if (!job)
      return job.takeError();
    auto key =
        implementationKey(*job, configurationAbi.abi(), implementationPlatform);
    if (!key)
      return key.takeError();
  }
  preflight.finish(module);

  FabricOperationProviderOutput output;
  std::vector<AppliedSpecialization> applied;
  applied.reserve(orderedLeaves.size());
  std::map<std::vector<std::uint8_t>, SharedProviderImplementation>
      implementations;
  bool committed = false;
  llvm::scope_exit transactionOutcome([&] {
    RtlMaterializationStageTracker outcome(
        committed ? "specialization_commit" : "specialization_rollback",
        materializationKey, module);
    if (!committed)
      rollbackSpecializations(applied);
    outcome.finish(module);
  });

  RtlMaterializationStageTracker leafTransactions(
      "specialization_leaf_prepare_validate_apply_release_intern",
      materializationKey, module);
  std::map<std::string, std::vector<circt::hw::InstanceOp>> instancesByModule;
  module.walk([&](circt::hw::InstanceOp instance) {
    instancesByModule[instance.getModuleName().str()].push_back(instance);
  });
  std::vector<circt::hw::HWModuleOp> operationShells;
  std::set<mlir::Operation *> seenShells;
  for (const OrderedOperationLeaf &ordered : orderedLeaves) {
    circt::hw::HWModuleGeneratedOp leaf = ordered.association->module;
    const std::string symbol = leaf.getSymName().str();
    const auto instances = instancesByModule.find(symbol);
    if (instances == instancesByModule.end())
      continue;
    for (circt::hw::InstanceOp instance : instances->second) {
      auto shell = instance->getParentOfType<circt::hw::HWModuleOp>();
      if (shell && seenShells.insert(shell.getOperation()).second)
        operationShells.push_back(shell);
    }
  }
  llvm::sort(operationShells,
             [](circt::hw::HWModuleOp lhs, circt::hw::HWModuleOp rhs) {
               return lhs.getSymName() < rhs.getSymName();
             });
  const std::vector<circt::hw::HWModuleOp> fuModules =
      parentModulesOf(operationShells, instancesByModule);
  const std::vector<circt::hw::HWModuleOp> peModules =
      parentModulesOf(fuModules, instancesByModule);
  for (const OrderedOperationLeaf &ordered : orderedLeaves) {
    auto job = resolveJob(ordered);
    if (!job)
      return job.takeError();
    auto key =
        implementationKey(*job, configurationAbi.abi(), implementationPlatform);
    if (!key)
      return key.takeError();

    auto shared = implementations.find(*key);
    if (shared != implementations.end()) {
      const auto instances = instancesByModule.find(job->leafSymbol);
      auto change = applySharedImplementationReference(
          module, *job, shared->second.symbol,
          instances == instancesByModule.end()
              ? llvm::ArrayRef<circt::hw::InstanceOp>{}
              : llvm::ArrayRef(instances->second));
      if (!change)
        return change.takeError();
      applied.push_back(std::move(*change));
      FabricOperationProviderOutput relations =
          occurrenceRelationsForReuse(shared->second, job->occurrence);
      if (llvm::Error error = appendOutput(output, std::move(relations)))
        return std::move(error);
      continue;
    }

    auto specialization = prepareSpecialization(*job, configurationAbi.abi(),
                                                implementationPlatform);
    if (!specialization)
      return specialization.takeError();
    if (llvm::Error error = validateSymbolClosure(module, *specialization))
      return std::move(error);

    SharedProviderImplementation implementation{
        job->leafSymbol, job->occurrence, specialization->output.activityPoints,
        specialization->output.externalImplementationBindings};
    auto change = applySpecialization(module, *specialization);
    if (!change)
      return change.takeError();
    applied.push_back(std::move(*change));
    if (llvm::Error error =
            appendOutput(output, std::move(specialization->output)))
      return std::move(error);
    implementations.emplace(std::move(*key), std::move(implementation));
  }
  leafTransactions.finish(module);

  {
    RtlMaterializationStageTracker shellIntern(
        "specialization_operation_shell_intern", materializationKey, module);
    if (llvm::Error error =
            internInternalModules(operationShells, instancesByModule, applied))
      return std::move(error);
    shellIntern.finish(module);
  }

  {
    RtlMaterializationStageTracker fuIntern("specialization_fu_module_intern",
                                            materializationKey, module);
    if (llvm::Error error =
            internInternalModules(fuModules, instancesByModule, applied))
      return std::move(error);
    fuIntern.finish(module);
  }

  {
    RtlMaterializationStageTracker peIntern("specialization_pe_module_intern",
                                            materializationKey, module);
    if (llvm::Error error =
            internInternalModules(peModules, instancesByModule, applied))
      return std::move(error);
    peIntern.finish(module);
  }

  {
    RtlMaterializationStageTracker finalVerify(
        "specialization_symbol_final_verify", materializationKey, module);
    if (llvm::Error error = verifySpecializedCirctModule(module))
      return std::move(error);
    finalVerify.finish(module);
  }
  committed = true;
  applied.clear();
  return output;
}

} // namespace loom::hardware::rtl
