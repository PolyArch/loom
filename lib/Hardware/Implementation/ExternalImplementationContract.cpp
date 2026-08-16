#include "Hardware/Implementation/HardwareImplementation.h"

#include "HardwareImplementationInternal.h"

#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>

namespace loom::hardware {
namespace {

using ByteVector = std::vector<std::uint8_t>;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "external_implementation_contract_invalid: " +
                                     message);
}

llvm::Error validateKey(llvm::StringRef value, llvm::StringRef field) {
  if (value.empty())
    return invalid(field + " must be nonempty");
  const auto allowed = [](char character) {
    const unsigned char byte = static_cast<unsigned char>(character);
    return std::isalnum(byte) || character == '.' || character == '_' ||
           character == '-' || character == ':' || character == '/' ||
           character == '@';
  };
  if (!std::isalnum(static_cast<unsigned char>(value.front())) ||
      !std::isalnum(static_cast<unsigned char>(value.back())) ||
      !llvm::all_of(value, allowed))
    return invalid(field + " is not a canonical ASCII key");
  return llvm::Error::success();
}

bool validDependencyKind(ExternalDependencyKind kind) {
  return kind == ExternalDependencyKind::ExplicitFile ||
         kind == ExternalDependencyKind::ToolBundledResource;
}

bool validRepresentation(RepresentationRootVariant representation) {
  switch (representation) {
  case RepresentationRootVariant::Rtl:
  case RepresentationRootVariant::GateNetlist:
  case RepresentationRootVariant::AsicPhysical:
  case RepresentationRootVariant::FpgaPhysical:
  case RepresentationRootVariant::FpgaImage:
    return true;
  case RepresentationRootVariant::FabricModel:
    return false;
  }
  return false;
}

ExternalDependencyKind
dependencyKind(const ExternalDependencyIdentity &identity) {
  return std::holds_alternative<ExplicitFileDependency>(identity)
             ? ExternalDependencyKind::ExplicitFile
             : ExternalDependencyKind::ToolBundledResource;
}

bool dependencyLess(const ExternalDependencyIdentity &lhs,
                    const ExternalDependencyIdentity &rhs) {
  if (dependencyKind(lhs) != dependencyKind(rhs))
    return dependencyKind(lhs) < dependencyKind(rhs);
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

bool externalInputLess(const ExternalInputBinding &lhs,
                       const ExternalInputBinding &rhs) {
  if (lhs.providerInputSlotRef != rhs.providerInputSlotRef)
    return lhs.providerInputSlotRef < rhs.providerInputSlotRef;
  return dependencyLess(lhs.dependencyIdentity, rhs.dependencyIdentity);
}

bool locatorLess(const RepresentationLocator &lhs,
                 const RepresentationLocator &rhs) {
  return std::tie(lhs.kind, lhs.canonicalName) <
         std::tie(rhs.kind, rhs.canonicalName);
}

bool physicalOwnerLess(const fabric::FabricPhysicalOccurrenceOwnerRef &lhs,
                       const fabric::FabricPhysicalOccurrenceOwnerRef &rhs) {
  return fabric::canonicalFabricBytes(lhs) < fabric::canonicalFabricBytes(rhs);
}

llvm::Error
validateInputsAgainstContract(std::vector<ExternalInputBinding> &inputs,
                              const ExternalImplementationContract &contract,
                              RepresentationRootVariant representation) {
  if (!llvm::is_contained(contract.supportedRepresentations, representation))
    return invalid("provider contract does not support the representation");

  llvm::sort(inputs, externalInputLess);
  if (inputs.size() != contract.inputSlots.size())
    return invalid("provider input slot closure is incomplete");
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    ExternalInputBinding &input = inputs[index];
    const ExternalInputSlotContract &slot = contract.inputSlots[index];
    if (input.providerInputSlotRef != slot.providerInputSlotRef)
      return invalid("provider input slot closure does not match contract");
    if (!llvm::is_contained(slot.acceptedDependencyKinds,
                            dependencyKind(input.dependencyIdentity)))
      return invalid("provider input dependency kind is incompatible");
    if (const auto *bundled = std::get_if<ToolBundledResourceDependency>(
            &input.dependencyIdentity)) {
      if (llvm::Error error = validateKey(bundled->stableProviderBuildIdentity,
                                          "stable provider build identity"))
        return error;
      if (llvm::Error error =
              validateKey(bundled->resourceKey, "provider resource key"))
        return error;
    }
  }
  return llvm::Error::success();
}

bool externalBindingLess(const ExternalImplementationBindingDraft &lhs,
                         const ExternalImplementationBindingDraft &rhs) {
  if (lhs.providerContractRef != rhs.providerContractRef)
    return lhs.providerContractRef < rhs.providerContractRef;
  if (lhs.externalInputs != rhs.externalInputs)
    return std::lexicographical_compare(
        lhs.externalInputs.begin(), lhs.externalInputs.end(),
        rhs.externalInputs.begin(), rhs.externalInputs.end(),
        externalInputLess);
  if (lhs.fabricResourceRefs != rhs.fabricResourceRefs)
    return std::lexicographical_compare(
        lhs.fabricResourceRefs.begin(), lhs.fabricResourceRefs.end(),
        rhs.fabricResourceRefs.begin(), rhs.fabricResourceRefs.end(),
        physicalOwnerLess);
  if (lhs.representationLocators != rhs.representationLocators)
    return std::lexicographical_compare(
        lhs.representationLocators.begin(), lhs.representationLocators.end(),
        rhs.representationLocators.begin(), rhs.representationLocators.end(),
        locatorLess);
  if (lhs.blackBoxContractPayload.has_value() !=
      rhs.blackBoxContractPayload.has_value())
    return !lhs.blackBoxContractPayload;
  if (!lhs.blackBoxContractPayload)
    return false;
  return std::tie(lhs.blackBoxContractPayload->role,
                  lhs.blackBoxContractPayload->canonicalLogicalName) <
         std::tie(rhs.blackBoxContractPayload->role,
                  rhs.blackBoxContractPayload->canonicalLogicalName);
}

bool sameExternalBinding(const ExternalImplementationBindingDraft &lhs,
                         const ExternalImplementationBindingDraft &rhs) {
  return !externalBindingLess(lhs, rhs) && !externalBindingLess(rhs, lhs);
}

} // namespace

llvm::Error ExternalImplementationContractCatalog::add(
    ExternalImplementationContract contract) {
  if (llvm::Error error = validateKey(contract.contractRef, "contract ref"))
    return error;
  if (contract.inputSlots.empty())
    return invalid("input slot catalog must be nonempty");
  for (ExternalInputSlotContract &slot : contract.inputSlots) {
    if (llvm::Error error =
            validateKey(slot.providerInputSlotRef, "provider input slot ref"))
      return error;
    if (slot.acceptedDependencyKinds.empty())
      return invalid("accepted dependency kind catalog must be nonempty");
    llvm::sort(slot.acceptedDependencyKinds);
    if (!llvm::all_of(slot.acceptedDependencyKinds, validDependencyKind) ||
        std::adjacent_find(slot.acceptedDependencyKinds.begin(),
                           slot.acceptedDependencyKinds.end()) !=
            slot.acceptedDependencyKinds.end())
      return invalid("accepted dependency kinds are invalid or duplicated");
  }
  llvm::sort(contract.inputSlots, [](const ExternalInputSlotContract &lhs,
                                     const ExternalInputSlotContract &rhs) {
    return lhs.providerInputSlotRef < rhs.providerInputSlotRef;
  });
  for (std::size_t index = 1; index < contract.inputSlots.size(); ++index)
    if (contract.inputSlots[index - 1].providerInputSlotRef ==
        contract.inputSlots[index].providerInputSlotRef)
      return invalid("provider input slot catalog contains a duplicate");

  if (contract.supportedRepresentations.empty())
    return invalid("supported representation catalog must be nonempty");
  llvm::sort(contract.supportedRepresentations);
  if (!llvm::all_of(contract.supportedRepresentations, validRepresentation) ||
      std::adjacent_find(contract.supportedRepresentations.begin(),
                         contract.supportedRepresentations.end()) !=
          contract.supportedRepresentations.end())
    return invalid("supported representations are invalid or duplicated");

  auto insertion = llvm::lower_bound(
      contracts_, contract.contractRef,
      [](const ExternalImplementationContract &entry, llvm::StringRef key) {
        return entry.contractRef < key;
      });
  if (insertion != contracts_.end() &&
      insertion->contractRef == contract.contractRef)
    return invalid("provider contract ref is duplicated");
  contracts_.insert(insertion, std::move(contract));
  return llvm::Error::success();
}

std::optional<ExternalImplementationContract>
ExternalImplementationContractCatalog::find(llvm::StringRef contractRef) const {
  auto found = llvm::lower_bound(
      contracts_, contractRef,
      [](const ExternalImplementationContract &entry, llvm::StringRef key) {
        return entry.contractRef < key;
      });
  if (found == contracts_.end() || found->contractRef != contractRef)
    return std::nullopt;
  return *found;
}

llvm::Expected<std::vector<ExternalInputBinding>>
ExternalImplementationContractCatalog::canonicalizeAndValidateInputs(
    llvm::StringRef contractRef,
    llvm::ArrayRef<ExternalInputBinding> externalInputs,
    RepresentationRootVariant representation) const {
  std::optional<ExternalImplementationContract> contract = find(contractRef);
  if (!contract)
    return invalid("provider contract is not registered: " + contractRef);
  std::vector<ExternalInputBinding> canonicalInputs(externalInputs.begin(),
                                                    externalInputs.end());
  if (llvm::Error error = validateInputsAgainstContract(
          canonicalInputs, *contract, representation))
    return std::move(error);
  return canonicalInputs;
}

llvm::Error
ExternalImplementationContractCatalog::canonicalizeAndValidateBindings(
    std::vector<ExternalImplementationBindingDraft> &bindings,
    const ImplementationRepresentationRoot &representation,
    const platform::ImplementationPlatform *implementationPlatform,
    const fabric::FabricSystemRootView &fabric) const {
  for (ExternalImplementationBindingDraft &binding : bindings) {
    std::optional<ExternalImplementationContract> contract =
        find(binding.providerContractRef);
    if (!contract)
      return invalid("provider contract is not registered: " +
                     binding.providerContractRef);
    if (llvm::Error error = validateInputsAgainstContract(
            binding.externalInputs, *contract, representation.variant))
      return error;

    llvm::sort(binding.fabricResourceRefs, physicalOwnerLess);
    if (std::adjacent_find(binding.fabricResourceRefs.begin(),
                           binding.fabricResourceRefs.end()) !=
        binding.fabricResourceRefs.end())
      return invalid("external Fabric resource reference is duplicated");
    for (const fabric::FabricPhysicalOccurrenceOwnerRef &reference :
         binding.fabricResourceRefs) {
      auto resolved = fabric.resolvePhysicalOwner(reference);
      if (!resolved)
        return resolved.takeError();
    }

    llvm::sort(binding.representationLocators, locatorLess);
    if (std::adjacent_find(binding.representationLocators.begin(),
                           binding.representationLocators.end()) !=
        binding.representationLocators.end())
      return invalid("external representation locator is duplicated");
    for (const RepresentationLocator &locator : binding.representationLocators)
      if (llvm::Error error =
              detail::validateRepresentationLocator(locator, representation))
        return error;

    if (contract->blackBoxContractRequired && !binding.blackBoxContractPayload)
      return invalid("provider contract requires a BlackBoxContract payload");
    if (binding.blackBoxContractPayload &&
        binding.blackBoxContractPayload->role != PayloadRole::BlackBoxContract)
      return invalid("external payload reference is not a BlackBoxContract");

    if (contract->validator)
      if (llvm::Error error = contract->validator(binding, representation,
                                                  implementationPlatform))
        return error;
  }
  llvm::sort(bindings, externalBindingLess);
  bindings.erase(
      std::unique(bindings.begin(), bindings.end(), sameExternalBinding),
      bindings.end());
  return llvm::Error::success();
}

llvm::Expected<std::vector<MemoryMacroBinding>>
detail::canonicalizeMemoryMacroBindings(
    llvm::ArrayRef<MemoryMacroBindingDraft> bindings,
    llvm::ArrayRef<ExternalImplementationBinding> externalBindings,
    llvm::ArrayRef<std::uint64_t> authoredToCanonicalBinding,
    const ExternalImplementationContractCatalog &contracts,
    const ImplementationRepresentationRoot &representation,
    const fabric::FabricSystemRootView &fabric) {
  std::vector<MemoryMacroBinding> canonical;
  canonical.reserve(bindings.size());
  std::set<ByteVector> memories;
  for (const MemoryMacroBindingDraft &binding : bindings) {
    auto resolved = fabric.resolvePhysicalOwner(binding.fabricMemoryRef);
    if (!resolved)
      return resolved.takeError();
    if (resolved->localOwner.kind() !=
        fabric::FabricInventoryOwnerKind::MemoryOccurrence)
      return invalid(
          "memory macro Fabric reference is not a memory occurrence");
    ByteVector key = fabric::canonicalFabricBytes(binding.fabricMemoryRef);
    if (!memories.insert(key).second)
      return invalid("memory macro binding is duplicated");
    if (binding.externalImplementationBindingDraftIndex >=
        authoredToCanonicalBinding.size())
      return invalid("memory macro external binding draft index is unresolved");
    const std::uint64_t ordinal =
        authoredToCanonicalBinding[static_cast<std::size_t>(
            binding.externalImplementationBindingDraftIndex)];
    if (ordinal >= externalBindings.size())
      return invalid("memory macro external binding reference is unresolved");
    const ExternalImplementationBinding &external =
        externalBindings[static_cast<std::size_t>(ordinal)];
    std::optional<ExternalImplementationContract> contract =
        contracts.find(external.providerContractRef);
    if (!contract || !contract->memoryMacroCapable)
      return invalid("memory macro provider contract is not memory-capable");
    if (!llvm::is_contained(external.fabricResourceRefs,
                            binding.fabricMemoryRef))
      return invalid("memory macro Fabric reference is absent from its "
                     "external binding");
    if (!llvm::is_contained(external.representationLocators,
                            binding.representationLocator))
      return invalid(
          "memory macro locator is absent from its external binding");
    if (llvm::Error error = validateRepresentationLocator(
            binding.representationLocator, representation))
      return std::move(error);
    canonical.push_back(MemoryMacroBinding{
        binding.fabricMemoryRef, ExternalImplementationBindingRef{ordinal},
        binding.representationLocator});
  }
  llvm::sort(canonical,
             [](const MemoryMacroBinding &lhs, const MemoryMacroBinding &rhs) {
               return fabric::canonicalFabricBytes(lhs.fabricMemoryRef) <
                      fabric::canonicalFabricBytes(rhs.fabricMemoryRef);
             });
  return canonical;
}

} // namespace loom::hardware
