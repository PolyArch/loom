#include "Hardware/Implementation/FpgaNativeExternalContracts.h"

#include "Common/BlobDigest.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <cstdint>
#include <variant>
#include <vector>

namespace loom::hardware {
namespace {

constexpr llvm::StringLiteral kAmdVivadoBuild =
    "SW Build 6511674 on Tue Jun 16 11:01:26 MDT 2026";
constexpr llvm::StringLiteral kAmdProviderIdentity =
    "amd_vivado_build_5357204275696c642036353131363734206f6e20547565204a75"
    "6e2031362031313a30313a3236204d44542032303236";
constexpr llvm::StringLiteral kAmdBlackBoxContract =
    "{\"contract\":\"amd.xilinx.unisim.dsp58@2\","
    "\"device\":\"xcvp1802-vsva5601-3HP-e-S\","
    "\"latency\":\"combinational\",\"module\":\"DSP58\","
    "\"operation\":\"i16_mul_mod\","
    "\"resource\":\"unisim:versal:DSP58\","
    "\"tool_build\":\"amd_vivado_build_5357204275696c642036353131363734"
    "206f6e20547565204a756e2031362031313a30313a3236204d44542032303236\"}\n";

constexpr llvm::StringLiteral kIntelBlackBoxContract =
    "{\"contract\":\"intel.altera.lpm_mult@1\","
    "\"device\":\"AGIA040R39A1E1VC\","
    "\"latency\":\"combinational\",\"module\":\"lpm_mult\","
    "\"operation\":\"i16_mul_mod\","
    "\"parameters\":\"widtha:16,widthb:16,widthp:32,pipeline:0,"
    "representation:UNSIGNED,dedicated_multiplier:YES\","
    "\"ports\":\"dataa:input:16,datab:input:16,result:output:32\","
    "\"resource\":\"altera_lpm:lpm_mult\","
    "\"tool_build\":\"altera.quartus-prime-pro:26.1.0-build-110\"}\n";

const FpgaNativeExternalModuleContract kAmdContract{
    platform::FpgaVendor::AmdXilinx,
    "amd.xilinx.unisim.dsp58@2",
    "primitive",
    kAmdProviderIdentity,
    "unisim:versal:DSP58",
    "xcvp1802-vsva5601-3HP-e-S",
    "DSP58",
    "contracts/amd_xilinx_unisim_dsp58.json",
    kAmdBlackBoxContract};

const FpgaNativeExternalModuleContract kIntelContract{
    platform::FpgaVendor::IntelAltera,
    "intel.altera.lpm_mult@1",
    "configured_ip",
    "altera.quartus-prime-pro:26.1.0-build-110",
    "altera_lpm:lpm_mult",
    "AGIA040R39A1E1VC",
    "lpm_mult",
    "contracts/intel_altera_lpm_mult_i16.json",
    kIntelBlackBoxContract};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fpga_native_external_contract_invalid: " +
                                     message);
}

bool isExactPlatform(const FpgaNativeExternalModuleContract &definition,
                     const platform::ImplementationPlatform *platform) {
  if (!platform)
    return false;
  const auto *target = std::get_if<platform::FpgaTarget>(&platform->target());
  return target && target->vendor == definition.vendor &&
         target->deviceOrderingCode == definition.deviceOrderingCode;
}

bool isExactExternalInput(const FpgaNativeExternalModuleContract &definition,
                          llvm::ArrayRef<ExternalInputBinding> inputs) {
  if (inputs.size() != 1 ||
      inputs.front().providerInputSlotRef != definition.providerInputSlotRef)
    return false;
  const auto *resource = std::get_if<ToolBundledResourceDependency>(
      &inputs.front().dependencyIdentity);
  return resource &&
         resource->stableProviderBuildIdentity ==
             definition.stableProviderBuildIdentity &&
         resource->resourceKey == definition.resourceKey;
}

llvm::Error
validateBinding(const FpgaNativeExternalModuleContract &definition,
                const ExternalImplementationBindingDraft &binding,
                const ImplementationRepresentationRoot &representation,
                const platform::ImplementationPlatform *platform) {
  const ImplementationPayload expectedPayload{
      PayloadRole::BlackBoxContract,
      definition.blackBoxPayloadLogicalName.str(),
      computeBlobDigest(llvm::ArrayRef<std::uint8_t>(
          definition.blackBoxContractBytes.bytes_begin(),
          definition.blackBoxContractBytes.bytes_end()))};
  if ((representation.variant != RepresentationRootVariant::Rtl &&
       representation.variant != RepresentationRootVariant::FpgaPhysical) ||
      !isExactPlatform(definition, platform) ||
      binding.providerContractRef != definition.contractRef ||
      !isExactExternalInput(definition, binding.externalInputs) ||
      binding.fabricResourceRefs.empty() ||
      binding.representationLocators !=
          std::vector<RepresentationLocator>{{RepresentationObjectKind::Module,
                                              definition.moduleName.str()}} ||
      !binding.blackBoxContractPayload ||
      !(*binding.blackBoxContractPayload ==
        ImplementationPayloadKey{
            PayloadRole::BlackBoxContract,
            definition.blackBoxPayloadLogicalName.str()}) ||
      !llvm::is_contained(representation.payloads, expectedPayload))
    return invalid("binding does not preserve the exact built-in FPGA native "
                   "module closure");
  return llvm::Error::success();
}

llvm::Error
validateAmdBinding(const ExternalImplementationBindingDraft &binding,
                   const ImplementationRepresentationRoot &representation,
                   const platform::ImplementationPlatform *platform) {
  return validateBinding(kAmdContract, binding, representation, platform);
}

llvm::Error
validateIntelBinding(const ExternalImplementationBindingDraft &binding,
                     const ImplementationRepresentationRoot &representation,
                     const platform::ImplementationPlatform *platform) {
  return validateBinding(kIntelContract, binding, representation, platform);
}

llvm::Error addContract(ExternalImplementationContractCatalog &catalog,
                        const FpgaNativeExternalModuleContract &definition,
                        ExternalImplementationBindingValidator validator) {
  return catalog.add(ExternalImplementationContract{
      definition.contractRef.str(),
      {{definition.providerInputSlotRef.str(),
        {ExternalDependencyKind::ToolBundledResource}}},
      {RepresentationRootVariant::Rtl, RepresentationRootVariant::FpgaPhysical},
      true,
      false,
      validator});
}

} // namespace

std::string amdVivadoToolBundledResourceProviderIdentity(
    llvm::StringRef stableProviderBuildIdentity) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string result = "amd_vivado_build_";
  result.reserve(result.size() + stableProviderBuildIdentity.size() * 2);
  for (const unsigned char byte : stableProviderBuildIdentity.bytes()) {
    result.push_back(kHex[byte >> 4]);
    result.push_back(kHex[byte & 0x0f]);
  }
  return result;
}

const FpgaNativeExternalModuleContract &amdXilinxDsp58ExternalModuleContract() {
  return kAmdContract;
}

const FpgaNativeExternalModuleContract &
intelAlteraLpmMultExternalModuleContract() {
  return kIntelContract;
}

llvm::Expected<ExternalImplementationContractCatalog>
makeFpgaNativeExternalImplementationContractCatalog() {
  if (amdVivadoToolBundledResourceProviderIdentity(kAmdVivadoBuild) !=
      kAmdProviderIdentity)
    return invalid("AMD provider identity constant is not derived from its "
                   "exact Vivado build");
  ExternalImplementationContractCatalog catalog;
  if (llvm::Error error =
          addContract(catalog, kAmdContract, validateAmdBinding))
    return std::move(error);
  if (llvm::Error error =
          addContract(catalog, kIntelContract, validateIntelBinding))
    return std::move(error);
  return catalog;
}

} // namespace loom::hardware
