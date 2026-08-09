#include "Hardware/Implementation/SynopsysDesignWareExternalContract.h"

#include "Common/BlobDigest.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <string>
#include <variant>

namespace loom::hardware {
namespace {

constexpr llvm::StringLiteral blackBoxContract =
    "synopsys.designware.DW_fp_mac.f32.rne.ieee.v1\n";

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "synopsys_designware_invalid: " + message);
}

llvm::Error
validateBinding(const ExternalImplementationBindingDraft &binding,
                const ImplementationRepresentationRoot &representation,
                const platform::ImplementationPlatform *) {
  if (representation.variant != RepresentationRootVariant::Rtl)
    return invalid("binding representation is not RTL");
  if (binding.providerContractRef != synopsysDesignWareContractRef ||
      !isSynopsysDesignWareDwFpMacComponentInput(binding.externalInputs))
    return invalid("binding does not select the verified component resource");
  if (binding.fabricResourceRefs.empty())
    return invalid("binding owns no physical occurrence");
  if (binding.representationLocators !=
      std::vector<RepresentationLocator>{
          {RepresentationObjectKind::Module,
           synopsysDesignWareDwFpMacComponentName.str()}})
    return invalid("binding does not locate the exact component module");
  if (!binding.blackBoxContractPayload ||
      !(*binding.blackBoxContractPayload ==
        ImplementationPayloadKey{
            PayloadRole::BlackBoxContract,
            synopsysDesignWareDwFpMacBlackBoxLogicalName.str()}))
    return invalid("binding does not select the exact BlackBoxContract");
  const ImplementationPayload expected{
      PayloadRole::BlackBoxContract,
      synopsysDesignWareDwFpMacBlackBoxLogicalName.str(),
      computeBlobDigest(synopsysDesignWareDwFpMacBlackBoxContractBytes())};
  if (!llvm::is_contained(representation.payloads, expected))
    return invalid("representation omits the exact BlackBoxContract payload");
  return llvm::Error::success();
}

} // namespace

llvm::ArrayRef<std::uint8_t> synopsysDesignWareDwFpMacBlackBoxContractBytes() {
  return llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(blackBoxContract.data()),
      blackBoxContract.size());
}

bool isSynopsysDesignWareDwFpMacComponentInput(
    llvm::ArrayRef<ExternalInputBinding> inputs) {
  if (inputs.size() != 1 || inputs.front().providerInputSlotRef !=
                                synopsysDesignWareComponentInputSlot)
    return false;
  const auto *resource = std::get_if<ToolBundledResourceDependency>(
      &inputs.front().dependencyIdentity);
  return resource &&
         resource->stableProviderBuildIdentity ==
             synopsysDesignWareBuildIdentity &&
         resource->resourceKey == synopsysDesignWareDwFpMacResourceKey;
}

llvm::Error registerSynopsysDesignWareExternalContract(
    ExternalImplementationContractCatalog &catalog) {
  return catalog.add({synopsysDesignWareContractRef.str(),
                      {{synopsysDesignWareComponentInputSlot.str(),
                        {ExternalDependencyKind::ToolBundledResource}}},
                      {RepresentationRootVariant::Rtl},
                      true,
                      false,
                      validateBinding});
}

} // namespace loom::hardware
