#include "Hardware/Implementation/FpgaNativeExternalContracts.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <vector>

namespace {

using namespace loom::hardware;
namespace platform = loom::platform;

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "fpgaNativeExternalContractsAreCanonical: " << message
               << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, llvm::StringRef message) {
  if (!condition)
    fail(message);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void checkCatalogEntry(const FpgaNativeExternalModuleContract &definition) {
  ExternalImplementationContractCatalog catalog =
      take(makeFpgaNativeExternalImplementationContractCatalog());
  auto contract = catalog.find(definition.contractRef);
  require(contract.has_value(), "catalog omitted a built-in contract");
  require(contract->inputSlots.size() == 1 &&
              contract->inputSlots.front().providerInputSlotRef ==
                  definition.providerInputSlotRef &&
              contract->inputSlots.front().acceptedDependencyKinds ==
                  std::vector<ExternalDependencyKind>{
                      ExternalDependencyKind::ToolBundledResource} &&
              contract->supportedRepresentations ==
                  std::vector<RepresentationRootVariant>{
                      RepresentationRootVariant::Rtl,
                      RepresentationRootVariant::FpgaPhysical} &&
              contract->blackBoxContractRequired &&
              !contract->memoryMacroCapable && contract->validator,
          "catalog changed the exact native FPGA contract shape");
}

void fpgaNativeExternalContractsAreCanonical() {
  const FpgaNativeExternalModuleContract &amd =
      amdXilinxDsp58ExternalModuleContract();
  const FpgaNativeExternalModuleContract &intel =
      intelAlteraLpmMultExternalModuleContract();
  require(amd.vendor == platform::FpgaVendor::AmdXilinx &&
              amd.contractRef == "amd.xilinx.unisim.dsp58@2" &&
              amd.providerInputSlotRef == "primitive" &&
              amd.resourceKey == "unisim:versal:DSP58" &&
              amd.deviceOrderingCode == "xcvp1802-vsva5601-3HP-e-S" &&
              amd.moduleName == "DSP58" &&
              amd.blackBoxPayloadLogicalName ==
                  "contracts/amd_xilinx_unisim_dsp58.json" &&
              amd.stableProviderBuildIdentity ==
                  amdVivadoToolBundledResourceProviderIdentity(
                      "SW Build 6511674 on Tue Jun 16 11:01:26 MDT 2026"),
          "AMD DSP58 contract facts diverged");
  require(intel.vendor == platform::FpgaVendor::IntelAltera &&
              intel.contractRef == "intel.altera.lpm_mult@1" &&
              intel.providerInputSlotRef == "configured_ip" &&
              intel.stableProviderBuildIdentity ==
                  "altera.quartus-prime-pro:26.1.0-build-110" &&
              intel.resourceKey == "altera_lpm:lpm_mult" &&
              intel.deviceOrderingCode == "AGIA040R39A1E1VC" &&
              intel.moduleName == "lpm_mult" &&
              intel.blackBoxPayloadLogicalName ==
                  "contracts/intel_altera_lpm_mult_i16.json",
          "Intel LPM_MULT contract facts diverged");
  require(!amd.blackBoxContractBytes.empty() &&
              !intel.blackBoxContractBytes.empty(),
          "built-in contract omitted its canonical payload bytes");
  checkCatalogEntry(amd);
  checkCatalogEntry(intel);
}

} // namespace

int main() {
  fpgaNativeExternalContractsAreCanonical();
  return EXIT_SUCCESS;
}
