#include "Hardware/Implementation/HardwareImplementation.h"

#include "HardwareImplementationInternal.h"

#include "Fabric/Artifact/FabricArtifactLocalReference.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

namespace loom::hardware {
namespace {

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

bool validRepresentation(HardwareRepresentation representation) {
  switch (representation) {
  case HardwareRepresentation::Rtl:
  case HardwareRepresentation::GateNetlist:
  case HardwareRepresentation::AsicPlaced:
  case HardwareRepresentation::AsicRouted:
  case HardwareRepresentation::AsicExtracted:
  case HardwareRepresentation::FpgaPlaced:
  case HardwareRepresentation::FpgaRouted:
  case HardwareRepresentation::FpgaImage:
    return true;
  }
  return false;
}

ExternalDependencyKind
dependencyKind(const ExternalDependencyIdentity &identity) {
  if (std::holds_alternative<ExplicitFileDependency>(identity))
    return ExternalDependencyKind::ExplicitFile;
  return ExternalDependencyKind::ToolBundledResource;
}

bool localReferenceLess(const EncodedArtifactLocalReference &lhs,
                        const EncodedArtifactLocalReference &rhs) {
  return encodeArtifactLocalReference(lhs) < encodeArtifactLocalReference(rhs);
}

bool locatorLess(const RepresentationLocator &lhs,
                 const RepresentationLocator &rhs) {
  return std::tie(lhs.kind, lhs.canonicalName) <
         std::tie(rhs.kind, rhs.canonicalName);
}

const HardwarePayload *findPayload(llvm::ArrayRef<HardwarePayload> payloads,
                                   const HardwarePayloadRef &reference) {
  auto found = llvm::find_if(payloads, [&](const HardwarePayload &payload) {
    return payload.role == reference.role &&
           payload.logicalName == reference.logicalName;
  });
  return found == payloads.end() ? nullptr : &*found;
}

llvm::Error
validateInputsAgainstContract(std::vector<ExternalInputBinding> &inputs,
                              const ExternalImplementationContract &contract,
                              HardwareRepresentation representation) {
  if (!llvm::is_contained(contract.supportedRepresentations, representation))
    return invalid("provider contract does not support the representation");

  llvm::sort(inputs, [](const ExternalInputBinding &lhs,
                        const ExternalInputBinding &rhs) {
    return lhs.providerInputSlotRef < rhs.providerInputSlotRef;
  });
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
    HardwareRepresentation representation) const {
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
    std::vector<ExternalImplementationBinding> &bindings,
    HardwareRepresentation representation,
    const platform::ImplementationPlatform *implementationPlatform,
    llvm::ArrayRef<HardwarePayload> payloads,
    const fabric::FabricArtifactView &fabric) const {
  llvm::sort(bindings, [](const ExternalImplementationBinding &lhs,
                          const ExternalImplementationBinding &rhs) {
    return lhs.bindingId < rhs.bindingId;
  });
  for (std::size_t bindingIndex = 0; bindingIndex < bindings.size();
       ++bindingIndex) {
    ExternalImplementationBinding &binding = bindings[bindingIndex];
    if (bindingIndex != 0 &&
        bindings[bindingIndex - 1].bindingId == binding.bindingId)
      return invalid("external implementation binding ID is duplicated");
    if (llvm::Error error = validateKey(binding.bindingId, "binding ID"))
      return error;

    std::optional<ExternalImplementationContract> contract =
        find(binding.providerContractRef);
    if (!contract)
      return invalid("provider contract is not registered: " +
                     binding.providerContractRef);
    if (llvm::Error error = validateInputsAgainstContract(
            binding.externalInputs, *contract, representation))
      return error;

    llvm::sort(binding.fabricResourceRefs, localReferenceLess);
    if (std::adjacent_find(binding.fabricResourceRefs.begin(),
                           binding.fabricResourceRefs.end()) !=
        binding.fabricResourceRefs.end())
      return invalid("external Fabric resource reference is duplicated");
    for (const EncodedArtifactLocalReference &reference :
         binding.fabricResourceRefs)
      if (llvm::Error error =
              fabric::validateFabricArtifactLocalReference(fabric, reference))
        return error;

    llvm::sort(binding.representationLocators, locatorLess);
    if (std::adjacent_find(binding.representationLocators.begin(),
                           binding.representationLocators.end()) !=
        binding.representationLocators.end())
      return invalid("external representation locator is duplicated");
    for (const RepresentationLocator &locator : binding.representationLocators)
      if (llvm::Error error =
              detail::validateRepresentationLocator(locator, representation))
        return error;

    if (contract->blackBoxContractRequired &&
        !binding.blackBoxContractPayloadRef)
      return invalid("provider contract requires a BlackBoxContract payload");
    if (binding.blackBoxContractPayloadRef) {
      if (binding.blackBoxContractPayloadRef->role !=
          HardwarePayloadRole::BlackBoxContract)
        return invalid("external payload reference is not a BlackBoxContract");
      if (!findPayload(payloads, *binding.blackBoxContractPayloadRef))
        return invalid("BlackBoxContract payload reference is unresolved");
    }

    if (contract->validator)
      if (llvm::Error error = contract->validator(binding, representation,
                                                  implementationPlatform))
        return error;
  }
  return llvm::Error::success();
}

llvm::Error detail::canonicalizeMemoryMacroBindings(
    std::vector<MemoryMacroBinding> &bindings,
    llvm::ArrayRef<ExternalImplementationBinding> externalBindings,
    const ExternalImplementationContractCatalog &contracts,
    HardwareRepresentation representation,
    const fabric::FabricArtifactView &fabric) {
  const auto memoryKey = [](const MemoryMacroBinding &binding) {
    return encodeArtifactLocalReference(
        fabric::encodeFabricArtifactLocalReference(binding.fabricMemoryRef));
  };
  llvm::sort(bindings,
             [&](const MemoryMacroBinding &lhs, const MemoryMacroBinding &rhs) {
               return memoryKey(lhs) < memoryKey(rhs);
             });
  for (std::size_t index = 0; index < bindings.size(); ++index) {
    MemoryMacroBinding &binding = bindings[index];
    if (index != 0 && memoryKey(bindings[index - 1]) == memoryKey(binding))
      return invalid("memory macro binding is duplicated");

    const EncodedArtifactLocalReference encodedMemory =
        fabric::encodeFabricArtifactLocalReference(binding.fabricMemoryRef);
    if (llvm::Error error =
            fabric::validateFabricArtifactLocalReference(fabric, encodedMemory))
      return error;
    if (llvm::Error error = validateKey(binding.externalImplementationBindingId,
                                        "external implementation binding ID"))
      return error;
    if (llvm::Error error = validateRepresentationLocator(
            binding.representationLocator, representation))
      return error;

    auto external = llvm::lower_bound(
        externalBindings, binding.externalImplementationBindingId,
        [](const ExternalImplementationBinding &candidate,
           llvm::StringRef bindingId) {
          return candidate.bindingId < bindingId;
        });
    if (external == externalBindings.end() ||
        external->bindingId != binding.externalImplementationBindingId)
      return invalid("memory macro external binding is unresolved");
    std::optional<ExternalImplementationContract> contract =
        contracts.find(external->providerContractRef);
    if (!contract || !contract->memoryMacroCapable)
      return invalid("memory macro provider contract is not memory-capable");
    if (!llvm::is_contained(external->fabricResourceRefs, encodedMemory))
      return invalid("memory macro Fabric reference is absent from its "
                     "external binding");
    if (!llvm::is_contained(external->representationLocators,
                            binding.representationLocator))
      return invalid(
          "memory macro locator is absent from its external binding");
  }
  return llvm::Error::success();
}

} // namespace loom::hardware
