#include "ExternalTool/Provider.h"

namespace loom::external_tool {
namespace {

ToolRuntimeCompatibility edaContainerCompatibility() {
  return ToolRuntimeCompatibility{true, {"almalinux9", "almalinux8"}};
}

} // namespace

const ExternalToolProviderDescriptor &polyArchContainerProvider() {
  static const ExternalToolProviderDescriptor provider{
      ToolProviderDescriptor{
          "polyarch_container",
          {"container"},
          {{"POLYARCH_CONTAINER_ROOT", "container"}},
          {"container"},
      },
      ToolVersionProbe{{"--version"}, "PolyArch container"},
      ToolRuntimeCompatibility{},
  };
  return provider;
}

const ExternalToolProviderDescriptor &verilatorProvider() {
  static const ExternalToolProviderDescriptor provider{
      ToolProviderDescriptor{
          "verilator",
          {"verilator"},
          {{"VERILATOR_ROOT", "bin/verilator"}},
          {"verilator"},
      },
      ToolVersionProbe{{"--version"}, "Verilator"},
      ToolRuntimeCompatibility{true, {"almalinux9", "almalinux8"}},
  };
  return provider;
}

const ExternalToolProviderDescriptor &yosysProvider() {
  static const ExternalToolProviderDescriptor provider{
      ToolProviderDescriptor{
          "yosys",
          {"yosys"},
          {{"YOSYS_ROOT", "bin/yosys"}},
          {"yosys"},
      },
      ToolVersionProbe{{"-V"}, "Yosys"},
      ToolRuntimeCompatibility{true, {"almalinux9", "almalinux8"}},
  };
  return provider;
}

const ExternalToolProviderDescriptor &openRoadProvider() {
  static const ExternalToolProviderDescriptor provider{
      ToolProviderDescriptor{
          "openroad",
          {"openroad"},
          {{"OPENROAD_ROOT", "bin/openroad"},
           {"OPENROAD_HOME", "bin/openroad"}},
          {"openroad/2026.08.06-b9a38929e342", "openroad"},
      },
      ToolVersionProbe{{"-version"}, "b9a38929e", {0}, "b9a38929e"},
      edaContainerCompatibility(),
  };
  return provider;
}

const ExternalToolProviderDescriptor &vcsProvider() {
  static const ExternalToolProviderDescriptor provider{
      ToolProviderDescriptor{
          "vcs",
          {"vcs"},
          {{"VCS_HOME", "bin/vcs"}, {"VCSMX_HOME", "bin/vcs"}},
          {"synopsys/vcs/Y-2026.03-SP1", "synopsys/vcs"},
      },
      ToolVersionProbe{
          {"-full64", "-ID"}, "Compiler version =", {0}, "Compiler version ="},
      edaContainerCompatibility(),
  };
  return provider;
}

const ExternalToolProviderDescriptor &designCompilerProvider() {
  static const ExternalToolProviderDescriptor provider{
      ToolProviderDescriptor{
          "dc_shell",
          {"dc_shell"},
          {{"DC_HOME", "bin/dc_shell"}, {"SYNOPSYS_HOME", "bin/dc_shell"}},
          {"synopsys/syn/Y-2026.03-SP2", "synopsys/syn"},
      },
      ToolVersionProbe{
          {"-version"}, "dc_shell version", {0, 1}, "dc_shell version"},
      edaContainerCompatibility(),
  };
  return provider;
}

const ExternalToolProviderDescriptor &fusionCompilerProvider() {
  static const ExternalToolProviderDescriptor provider{
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
  };
  return provider;
}

const ExternalToolProviderDescriptor &xceliumProvider() {
  static const ExternalToolProviderDescriptor provider{
      ToolProviderDescriptor{
          "xrun",
          {"xrun"},
          {{"XCELIUM_HOME", "tools.lnx86/inca/bin/64bit/xrun"},
           {"XRUN_HOME", "tools.lnx86/inca/bin/64bit/xrun"}},
          {"cadence/XCELIUM/2603", "cadence/XCELIUM"},
      },
      ToolVersionProbe{{"-version"}, "xrun", {0}, "xrun"},
      edaContainerCompatibility(),
  };
  return provider;
}

const ExternalToolProviderDescriptor &genusProvider() {
  static const ExternalToolProviderDescriptor provider{
      ToolProviderDescriptor{
          "genus",
          {"genus"},
          {{"DDI_HOME", "bin/genus"}, {"CDS_INST_DIR", "bin/genus"}},
          {"cadence/DDI/261", "cadence/DDI"},
      },
      ToolVersionProbe{
          {"-version"}, "Program Name: Genus", {0}, "Program Name: Genus"},
      edaContainerCompatibility(),
  };
  return provider;
}

const ExternalToolProviderDescriptor &innovusProvider() {
  static const ExternalToolProviderDescriptor provider{
      ToolProviderDescriptor{
          "innovus",
          {"innovus"},
          {{"DDI_HOME", "bin/innovus"}, {"CDS_INST_DIR", "bin/innovus"}},
          {"cadence/DDI/261", "cadence/DDI"},
      },
      ToolVersionProbe{
          {"-version"}, "@(#)CDS: Innovus", {0}, "@(#)CDS: Innovus"},
      edaContainerCompatibility(),
  };
  return provider;
}

const ExternalToolProviderDescriptor &joulesProvider() {
  static const ExternalToolProviderDescriptor provider{
      ToolProviderDescriptor{
          "joules",
          {"joules"},
          {{"JOULES_HOME", "bin/joules"},
           {"DDI_HOME", "bin/joules"},
           {"CDS_INST_DIR", "bin/joules"}},
          {"cadence/JOULES/261", "cadence/JOULES"},
      },
      ToolVersionProbe{
          {"-version"}, "Program Name: Joules", {0}, "Program Name: Joules"},
      edaContainerCompatibility(),
  };
  return provider;
}

const ExternalToolProviderDescriptor &tempusProvider() {
  static const ExternalToolProviderDescriptor provider{
      ToolProviderDescriptor{
          "tempus",
          {"tempus"},
          {{"TEMPUS_HOME", "bin/tempus"},
           {"SSV_HOME", "bin/tempus"},
           {"CDS_INST_DIR", "bin/tempus"}},
          {"cadence/TEMPUS/261", "cadence/TEMPUS"},
      },
      ToolVersionProbe{{"-version"}, "@(#)CDS: Tempus", {0}, "@(#)CDS: Tempus"},
      edaContainerCompatibility(),
  };
  return provider;
}

const ExternalToolProviderDescriptor &voltusProvider() {
  static const ExternalToolProviderDescriptor provider{
      ToolProviderDescriptor{
          "voltus",
          {"voltus"},
          {{"VOLTUS_HOME", "bin/voltus"},
           {"SSV_HOME", "bin/voltus"},
           {"CDS_INST_DIR", "bin/voltus"}},
          {"cadence/VOLTUS/261", "cadence/VOLTUS"},
      },
      ToolVersionProbe{{"-version"}, "@(#)CDS: Voltus", {0}, "@(#)CDS: Voltus"},
      edaContainerCompatibility(),
  };
  return provider;
}

const ExternalToolProviderDescriptor &vivadoProvider() {
  static const ExternalToolProviderDescriptor provider{
      ToolProviderDescriptor{
          "vivado",
          {"vivado"},
          {{"XILINX_VIVADO", "bin/vivado"}},
          {"amd/2026.1", "amd"},
      },
      ToolVersionProbe{{"-version"}, "vivado v", {0}, "vivado v"},
      edaContainerCompatibility(),
  };
  return provider;
}

const ExternalToolProviderDescriptor &quartusPrimeProvider() {
  static const ExternalToolProviderDescriptor provider{
      ToolProviderDescriptor{
          "quartus_sh",
          {"quartus_sh"},
          {{"QUARTUS_ROOTDIR", "bin/quartus_sh"},
           {"QUARTUS_ROOTDIR_OVERRIDE", "bin/quartus_sh"}},
          {"Altera/QuartusPrimePro/26.1", "Altera/QuartusPrimePro"},
      },
      ToolVersionProbe{{"--version"}, "Quartus Prime", {0}, "Version "},
      edaContainerCompatibility(),
  };
  return provider;
}

} // namespace loom::external_tool
