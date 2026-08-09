#include "ExternalTool/Provider.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace loom::external_tool;

namespace {

[[noreturn]] void fail(const std::string &message) {
  std::cerr << message << '\n';
  std::exit(1);
}

void require(bool condition, const std::string &message) {
  if (!condition)
    fail(message);
}

} // namespace

int main() {
  const ExternalToolProviderDescriptor &container = polyArchContainerProvider();
  require(container.binding.key == "polyarch_container" &&
              container.binding.executableNames ==
                  std::vector<std::string>{"container"} &&
              container.versionProbe.arguments ==
                  std::vector<std::string>{"--version"} &&
              container.versionProbe.requiredOutputSubstring ==
                  "PolyArch container",
          "PolyArch/container provider contract is incomplete");

  const ExternalToolProviderDescriptor &verilator = verilatorProvider();
  require(verilator.binding.key == "verilator" &&
              verilator.binding.environmentCandidates.front().variable ==
                  "VERILATOR_ROOT" &&
              verilator.binding.moduleAliases ==
                  std::vector<std::string>{"verilator"} &&
              verilator.runtimeCompatibility.supportsPolyArchContainer,
          "Verilator provider contract is incomplete");

  const ExternalToolProviderDescriptor &yosys = yosysProvider();
  require(yosys.binding.key == "yosys" &&
              yosys.versionProbe.arguments == std::vector<std::string>{"-V"} &&
              yosys.versionProbe.requiredOutputSubstring == "Yosys",
          "Yosys provider contract is incomplete");

  const ExternalToolProviderDescriptor &openroad = openRoadProvider();
  require(openroad.binding.key == "openroad" &&
              openroad.binding.executableNames ==
                  std::vector<std::string>{"openroad"} &&
              openroad.binding.environmentCandidates.size() == 2 &&
              openroad.binding.environmentCandidates[0].variable ==
                  "OPENROAD_ROOT" &&
              openroad.binding.environmentCandidates[0].relativeExecutable ==
                  "bin/openroad" &&
              openroad.binding.environmentCandidates[1].variable ==
                  "OPENROAD_HOME" &&
              openroad.binding.moduleAliases ==
                  std::vector<std::string>{"openroad/2026.08.06-b9a38929e342",
                                           "openroad"} &&
              openroad.versionProbe.arguments ==
                  std::vector<std::string>{"-version"} &&
              openroad.versionProbe.requiredOutputSubstring == "b9a38929e" &&
              openroad.versionProbe.selectedOutputLineSubstring == "b9a38929e",
          "OpenROAD provider contract is incomplete");

  const ExternalToolProviderDescriptor &vcs = vcsProvider();
  require(vcs.binding.key == "vcs" &&
              vcs.versionProbe.arguments ==
                  std::vector<std::string>{"-full64", "-ID"} &&
              vcs.versionProbe.selectedOutputLineSubstring ==
                  "Compiler version =" &&
              vcs.binding.moduleAliases.front() == "synopsys/vcs/Y-2026.03-SP1",
          "VCS provider contract is incomplete");

  const ExternalToolProviderDescriptor &dc = designCompilerProvider();
  require(dc.binding.key == "dc_shell" &&
              dc.versionProbe.acceptedExitCodes == std::vector<int>({0, 1}) &&
              dc.binding.moduleAliases.front() == "synopsys/syn/Y-2026.03-SP2",
          "Design Compiler provider contract is incomplete");

  const ExternalToolProviderDescriptor &fc = fusionCompilerProvider();
  require(fc.binding.key == "fc_shell" &&
              fc.binding.moduleAliases.front() ==
                  "synopsys/fusioncompiler/Y-2026.03",
          "Fusion Compiler provider contract is incomplete");

  const ExternalToolProviderDescriptor &xcelium = xceliumProvider();
  require(
      xcelium.binding.key == "xrun" &&
          xcelium.binding.environmentCandidates.front().variable ==
              "XCELIUM_HOME" &&
          xcelium.binding.environmentCandidates.front().relativeExecutable ==
              "tools.lnx86/inca/bin/64bit/xrun" &&
          xcelium.binding.moduleAliases.front() == "cadence/XCELIUM/2603",
      "Xcelium provider contract is incomplete");

  const ExternalToolProviderDescriptor &genus = genusProvider();
  const ExternalToolProviderDescriptor &innovus = innovusProvider();
  require(genus.binding.moduleAliases.front() == "cadence/DDI/261" &&
              innovus.binding.moduleAliases.front() == "cadence/DDI/261",
          "Cadence DDI provider contracts are incomplete");

  const ExternalToolProviderDescriptor &joules = joulesProvider();
  const ExternalToolProviderDescriptor &tempus = tempusProvider();
  const ExternalToolProviderDescriptor &voltus = voltusProvider();
  require(joules.binding.key == "joules" &&
              joules.binding.moduleAliases.front() == "cadence/JOULES/261" &&
              tempus.binding.key == "tempus" &&
              tempus.binding.moduleAliases.front() == "cadence/TEMPUS/261" &&
              voltus.binding.key == "voltus" &&
              voltus.binding.moduleAliases.front() == "cadence/VOLTUS/261",
          "Cadence evaluation provider contracts are incomplete");

  const ExternalToolProviderDescriptor &vivado = vivadoProvider();
  require(vivado.binding.key == "vivado" &&
              vivado.binding.moduleAliases.front() == "amd/2026.1",
          "Vivado provider must prefer the selected AMD 2026.1 suite");

  const ExternalToolProviderDescriptor &quartus = quartusPrimeProvider();
  require(quartus.binding.key == "quartus_sh" &&
              quartus.binding.moduleAliases.front() ==
                  "Altera/QuartusPrimePro/26.1",
          "Quartus Prime provider contract is incomplete");
  return 0;
}
