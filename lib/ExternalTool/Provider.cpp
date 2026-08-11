#include "ExternalTool/Provider.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <cassert>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace loom::external_tool {
namespace {

ToolRuntimeCompatibility edaContainerCompatibility() {
  return ToolRuntimeCompatibility{true, {"almalinux9", "almalinux8"}};
}

BackendToolReleaseProfile release(std::string feature, std::string moduleAlias,
                                  ToolVersionProbe exactVersionProbe) {
  return BackendToolReleaseProfile{std::move(feature), std::move(moduleAlias),
                                   std::move(exactVersionProbe)};
}

const std::vector<BackendToolCatalogEntry> &catalogStorage() {
  static const std::vector<BackendToolCatalogEntry> catalog{
      {"PolyArch Container Runtime",
       ExternalToolProviderDescriptor{
           ToolProviderDescriptor{
               "polyarch_container",
               {"container"},
               {{"POLYARCH_CONTAINER_ROOT", "container"}},
               {"container"},
           },
           ToolVersionProbe{{"--version"}, "PolyArch container"},
           ToolRuntimeCompatibility{},
       },
       {release("polyarch-container-0.1.0", "container",
                ToolVersionProbe{{"--version"},
                                 "PolyArch container v0.1.0",
                                 {0},
                                 "PolyArch container v0.1.0"})}},
      {"Verilator",
       ExternalToolProviderDescriptor{
           ToolProviderDescriptor{
               "verilator",
               {"verilator"},
               {{"VERILATOR_ROOT", "bin/verilator"}},
               {"verilator/5.050", "verilator"},
           },
           ToolVersionProbe{{"--version"}, "Verilator"},
           ToolRuntimeCompatibility{true, {"almalinux9", "almalinux8"}},
       },
       {release(
           "verilator", "verilator/5.050",
           ToolVersionProbe{
               {"--version"}, "Verilator 5.050", {0}, "Verilator 5.050"})}},
      {"Yosys",
       ExternalToolProviderDescriptor{
           ToolProviderDescriptor{
               "yosys",
               {"yosys"},
               {{"YOSYS_ROOT", "bin/yosys"}},
               {"yosys/0.67", "yosys"},
           },
           ToolVersionProbe{{"-V"}, "Yosys"},
           ToolRuntimeCompatibility{true, {"almalinux9", "almalinux8"}},
       },
       {release("yosys", "yosys/0.67",
                ToolVersionProbe{{"-V"}, "Yosys 0.67", {0}, "Yosys 0.67"})}},
      {"OpenROAD",
       ExternalToolProviderDescriptor{
           ToolProviderDescriptor{
               "openroad",
               {"openroad"},
               {{"OPENROAD_ROOT", "bin/openroad"},
                {"OPENROAD_HOME", "bin/openroad"}},
               {"openroad/2026.08.06-b9a38929e342", "openroad"},
           },
           ToolVersionProbe{{"-version"}, "b9a38929e", {0}, "b9a38929e"},
           edaContainerCompatibility(),
       },
       {release(
           "openroad-2026-08-06", "openroad/2026.08.06-b9a38929e342",
           ToolVersionProbe{{"-version"}, "b9a38929e", {0}, "b9a38929e"})}},
      {"gem5",
       ExternalToolProviderDescriptor{
           ToolProviderDescriptor{
               "gem5",
               {"gem5.opt"},
               {{"GEM5_ROOT", "build/RISCV/gem5.opt"}},
               {},
           },
           ToolVersionProbe{{"--build-info"}, "gem5 version"},
           ToolRuntimeCompatibility{},
       },
       {BackendToolReleaseProfile{
           "gem5-25.1.0.1", std::nullopt,
           ToolVersionProbe{{"--build-info"},
                            "gem5 version 25.1.0.1",
                            {0},
                            "gem5 version"}}}},
      {"Synopsys VCS",
       ExternalToolProviderDescriptor{
           ToolProviderDescriptor{
               "vcs",
               {"vcs"},
               {{"VCS_HOME", "bin/vcs"}, {"VCSMX_HOME", "bin/vcs"}},
               {"synopsys/vcs/Y-2026.03-SP1", "synopsys/vcs"},
           },
           ToolVersionProbe{{"-full64", "-ID"},
                            "Compiler version =",
                            {0},
                            "Compiler version ="},
           edaContainerCompatibility(),
       },
       {release("vcs-y-2026.03-sp1", "synopsys/vcs/Y-2026.03-SP1",
                ToolVersionProbe{{"-full64", "-ID"},
                                 "Y-2026.03-SP1_Full64",
                                 {0},
                                 "Compiler version ="})}},
      {"Synopsys Design Compiler",
       ExternalToolProviderDescriptor{
           ToolProviderDescriptor{
               "dc_shell",
               {"dc_shell"},
               {{"DC_HOME", "bin/dc_shell"}, {"SYNOPSYS_HOME", "bin/dc_shell"}},
               {"synopsys/syn/Y-2026.03-SP2", "synopsys/syn"},
           },
           ToolVersionProbe{
               {"-version"}, "dc_shell version", {0, 1}, "dc_shell version"},
           edaContainerCompatibility(),
       },
       {release(
           "design-compiler-y-2026.03-sp2", "synopsys/syn/Y-2026.03-SP2",
           ToolVersionProbe{
               {"-version"}, "Y-2026.03-SP2", {0, 1}, "dc_shell version"})}},
      {"Synopsys Fusion Compiler",
       ExternalToolProviderDescriptor{
           ToolProviderDescriptor{
               "fc_shell",
               {"fc_shell"},
               {{"FC_HOME", "bin/fc_shell"},
                {"FUSIONCOMPILER_HOME", "bin/fc_shell"}},
               {"synopsys/fusioncompiler/Y-2026.03", "synopsys/fusioncompiler"},
           },
           ToolVersionProbe{
               {"-version"}, "fc_shell version", {0}, "fc_shell version"},
           edaContainerCompatibility(),
       },
       {release("fusion-compiler-y-2026.03",
                "synopsys/fusioncompiler/Y-2026.03",
                ToolVersionProbe{
                    {"-version"}, "Y-2026.03", {0}, "fc_shell version"})}},
      {"Synopsys PrimeTime and PrimePower",
       ExternalToolProviderDescriptor{
           ToolProviderDescriptor{
               "pt_shell",
               {"pt_shell"},
               {{"PRIMETIME_HOME", "bin/pt_shell"},
                {"SYNOPSYS_HOME", "bin/pt_shell"}},
               {"synopsys/prime/Y-2026.03-SP2", "synopsys/prime"},
           },
           ToolVersionProbe{
               {"-version"}, "pt_shell version", {0, 1}, "pt_shell version"},
           edaContainerCompatibility(),
       },
       {release(
           "primetime-y-2026.03-sp2", "synopsys/prime/Y-2026.03-SP2",
           ToolVersionProbe{
               {"-version"}, "Y-2026.03-SP2", {0, 1}, "pt_shell version"})}},
      {"Cadence Xcelium",
       ExternalToolProviderDescriptor{
           ToolProviderDescriptor{
               "xrun",
               {"xrun"},
               {{"XCELIUM_HOME", "tools.lnx86/inca/bin/64bit/xrun"},
                {"XRUN_HOME", "tools.lnx86/inca/bin/64bit/xrun"}},
               {"cadence/XCELIUM/2603", "cadence/XCELIUM"},
           },
           ToolVersionProbe{{"-version"}, "xrun", {0}, "xrun"},
           edaContainerCompatibility(),
       },
       {release(
           "xcelium-26.03-s005", "cadence/XCELIUM/2603",
           ToolVersionProbe{{"-version"}, "26.03-s005", {0}, "xrun(64)"})}},
      {"Cadence Genus",
       ExternalToolProviderDescriptor{
           ToolProviderDescriptor{
               "genus",
               {"genus"},
               {{"DDI_HOME", "bin/genus"}, {"CDS_INST_DIR", "bin/genus"}},
               {"cadence/DDI/261", "cadence/DDI"},
           },
           ToolVersionProbe{
               {"-version"}, "Program Name: Genus", {0}, "Program Name: Genus"},
           edaContainerCompatibility(),
       },
       {release("genus-26.10-p002.1", "cadence/DDI/261",
                ToolVersionProbe{{"-version"},
                                 "Version: 26.10-p002_1",
                                 {0},
                                 "Program Name: Genus"})}},
      {"Cadence Innovus",
       ExternalToolProviderDescriptor{
           ToolProviderDescriptor{
               "innovus",
               {"innovus"},
               {{"DDI_HOME", "bin/innovus"}, {"CDS_INST_DIR", "bin/innovus"}},
               {"cadence/DDI/261", "cadence/DDI"},
           },
           ToolVersionProbe{
               {"-version"}, "@(#)CDS: Innovus", {0}, "@(#)CDS: Innovus"},
           edaContainerCompatibility(),
       },
       {release("innovus-26.10-p003.1", "cadence/DDI/261",
                ToolVersionProbe{
                    {"-version"}, "v26.10-p003_1", {0}, "@(#)CDS: Innovus"})}},
      {"Cadence Joules",
       ExternalToolProviderDescriptor{
           ToolProviderDescriptor{
               "joules",
               {"joules"},
               {{"JOULES_HOME", "bin/joules"},
                {"DDI_HOME", "bin/joules"},
                {"CDS_INST_DIR", "bin/joules"}},
               {"cadence/JOULES/261", "cadence/JOULES"},
           },
           ToolVersionProbe{{"-version"},
                            "Program Name: Joules",
                            {0},
                            "Program Name: Joules"},
           edaContainerCompatibility(),
       },
       {release("joules-26.10-p001.1", "cadence/JOULES/261",
                ToolVersionProbe{{"-version"},
                                 "Version: 26.10-p001_1",
                                 {0},
                                 "Program Name: Joules"})}},
      {"Cadence Tempus",
       ExternalToolProviderDescriptor{
           ToolProviderDescriptor{
               "tempus",
               {"tempus"},
               {{"TEMPUS_HOME", "bin/tempus"},
                {"SSV_HOME", "bin/tempus"},
                {"CDS_INST_DIR", "bin/tempus"}},
               {"cadence/TEMPUS/261", "cadence/TEMPUS"},
           },
           ToolVersionProbe{
               {"-version"}, "@(#)CDS: Tempus", {0}, "@(#)CDS: Tempus"},
           edaContainerCompatibility(),
       },
       {release("tempus-26.10-p001.1", "cadence/TEMPUS/261",
                ToolVersionProbe{
                    {"-version"}, "v26.10-p001_1", {0}, "@(#)CDS: Tempus"})}},
      {"Cadence Voltus",
       ExternalToolProviderDescriptor{
           ToolProviderDescriptor{
               "voltus",
               {"voltus"},
               {{"VOLTUS_HOME", "bin/voltus"},
                {"SSV_HOME", "bin/voltus"},
                {"CDS_INST_DIR", "bin/voltus"}},
               {"cadence/VOLTUS/261", "cadence/VOLTUS"},
           },
           ToolVersionProbe{
               {"-version"}, "@(#)CDS: Voltus", {0}, "@(#)CDS: Voltus"},
           edaContainerCompatibility(),
       },
       {release("voltus-26.10-p001.1", "cadence/VOLTUS/261",
                ToolVersionProbe{
                    {"-version"}, "v26.10-p001_1", {0}, "@(#)CDS: Voltus"})}},
      {"AMD Vivado",
       ExternalToolProviderDescriptor{
           ToolProviderDescriptor{
               "vivado",
               {"vivado"},
               {{"XILINX_VIVADO", "bin/vivado"}},
               {"amd/vivado/2024.2", "amd/vivado"},
           },
           ToolVersionProbe{{"-version"}, "vivado v", {0}, "vivado v"},
           edaContainerCompatibility(),
       },
       {release(
           "vivado-2024.2.2", "amd/vivado/2024.2",
           ToolVersionProbe{
               {"-version"}, "vivado v2024.2.2", {0}, "vivado v2024.2.2"})}},
      {"Altera Quartus Prime Pro",
       ExternalToolProviderDescriptor{
           ToolProviderDescriptor{
               "quartus_sh",
               {"quartus_sh"},
               {{"QUARTUS_ROOTDIR", "bin/quartus_sh"},
                {"QUARTUS_ROOTDIR_OVERRIDE", "bin/quartus_sh"}},
               {"Altera/QuartusPrimePro/26.1", "Altera/QuartusPrimePro"},
           },
           ToolVersionProbe{{"--version"}, "Quartus Prime", {0}, "Version "},
           edaContainerCompatibility(),
       },
       {release("quartus-prime-pro-26.1", "Altera/QuartusPrimePro/26.1",
                ToolVersionProbe{{"--version"},
                                 "Version 26.1.0 Build 110",
                                 {0},
                                 "Version 26.1.0 Build 110"})}},
  };
  return catalog;
}

llvm::Error catalogError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "backend_tool_catalog_invalid: " + message);
}

const ExternalToolProviderDescriptor &provider(llvm::StringRef key) {
  const BackendToolCatalogEntry *entry = findBackendTool(key);
  assert(entry && "catalog-owned provider is missing");
  return entry->provider;
}

} // namespace

llvm::ArrayRef<BackendToolCatalogEntry> backendToolCatalog() {
  return catalogStorage();
}

const BackendToolCatalogEntry *findBackendTool(llvm::StringRef logicalToolKey) {
  const auto &catalog = catalogStorage();
  const auto found = llvm::find_if(catalog, [&](const auto &entry) {
    return entry.provider.binding.key == logicalToolKey;
  });
  return found == catalog.end() ? nullptr : &*found;
}

llvm::Error validateBackendToolCatalog() {
  const auto &catalog = catalogStorage();
  if (catalog.empty())
    return catalogError("catalog is empty");
  std::set<std::string> keys;
  std::set<std::string> officialNames;
  std::set<std::string> features;
  for (const BackendToolCatalogEntry &entry : catalog) {
    if (entry.officialProductName.empty())
      return catalogError("official product name is empty");
    if (!officialNames.insert(entry.officialProductName).second)
      return catalogError("official product name is not unique: " +
                          entry.officialProductName);
    if (entry.provider.binding.key.empty() ||
        !keys.insert(entry.provider.binding.key).second)
      return catalogError("logical tool key is empty or not unique: " +
                          entry.provider.binding.key);
    if (entry.provider.binding.executableNames.empty())
      return catalogError("provider has no executable: " +
                          entry.provider.binding.key);
    for (const BackendToolReleaseProfile &profile : entry.validatedReleases) {
      if (profile.conformanceFeature.empty() ||
          !features.insert(profile.conformanceFeature).second)
        return catalogError("conformance feature is empty or not unique: " +
                            profile.conformanceFeature);
      if (profile.moduleAlias &&
          !llvm::is_contained(entry.provider.binding.moduleAliases,
                              *profile.moduleAlias))
        return catalogError("release module alias is not provider-owned: " +
                            *profile.moduleAlias);
      if (profile.exactVersionProbe.acceptedExitCodes.empty() ||
          !profile.exactVersionProbe.requiredOutputSubstring)
        return catalogError("release probe is not exact: " +
                            profile.conformanceFeature);
    }
  }
  return llvm::Error::success();
}

const ExternalToolProviderDescriptor &polyArchContainerProvider() {
  return provider("polyarch_container");
}

const ExternalToolProviderDescriptor &verilatorProvider() {
  return provider("verilator");
}

const ExternalToolProviderDescriptor &yosysProvider() {
  return provider("yosys");
}

const ExternalToolProviderDescriptor &openRoadProvider() {
  return provider("openroad");
}

const ExternalToolProviderDescriptor &gem5Provider() {
  return provider("gem5");
}

const ExternalToolProviderDescriptor &vcsProvider() { return provider("vcs"); }

const ExternalToolProviderDescriptor &designCompilerProvider() {
  return provider("dc_shell");
}

const ExternalToolProviderDescriptor &fusionCompilerProvider() {
  return provider("fc_shell");
}

const ExternalToolProviderDescriptor &primeTimeProvider() {
  return provider("pt_shell");
}

const ExternalToolProviderDescriptor &xceliumProvider() {
  return provider("xrun");
}

const ExternalToolProviderDescriptor &genusProvider() {
  return provider("genus");
}

const ExternalToolProviderDescriptor &innovusProvider() {
  return provider("innovus");
}

const ExternalToolProviderDescriptor &joulesProvider() {
  return provider("joules");
}

const ExternalToolProviderDescriptor &tempusProvider() {
  return provider("tempus");
}

const ExternalToolProviderDescriptor &voltusProvider() {
  return provider("voltus");
}

const ExternalToolProviderDescriptor &vivadoProvider() {
  return provider("vivado");
}

const ExternalToolProviderDescriptor &quartusPrimeProvider() {
  return provider("quartus_sh");
}

} // namespace loom::external_tool
