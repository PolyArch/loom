#include "OpenRoadPhysicalTestSupport.h"

#include "Common/ComponentViewDigest.h"
#include "EDA/Adapters/AsicStandardCellContracts.h"
#include "ExternalTool/Provider.h"
#include "Hardware/Configuration/ConfigurationABI.h"

#include "ConfigurationABITestSupport.h"

#include "ADG/Builder.h"

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <system_error>
#include <utility>

namespace loom::eda::open_source::test {
namespace {

llvm::Error failure(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "openroad_test_fixture: " + message);
}

std::vector<std::uint8_t> bytes(llvm::StringRef value) {
  return {value.bytes_begin(), value.bytes_end()};
}

constexpr llvm::StringLiteral kTechnologyLef = R"lef(VERSION 5.8 ;
BUSBITCHARS "[]" ;
DIVIDERCHAR "/" ;
UNITS
  DATABASE MICRONS 1000 ;
END UNITS
MANUFACTURINGGRID 0.001 ;
LAYER Metal1
  TYPE ROUTING ;
  DIRECTION VERTICAL ;
  PITCH 0.20 ;
  WIDTH 0.10 ;
  SPACING 0.10 ;
END Metal1
LAYER Via12
  TYPE CUT ;
  SPACING 0.10 ;
END Via12
LAYER Metal2
  TYPE ROUTING ;
  DIRECTION HORIZONTAL ;
  PITCH 0.20 ;
  WIDTH 0.10 ;
  SPACING 0.10 ;
END Metal2
LAYER Via23
  TYPE CUT ;
  SPACING 0.10 ;
END Via23
LAYER Metal3
  TYPE ROUTING ;
  DIRECTION VERTICAL ;
  PITCH 0.20 ;
  WIDTH 0.10 ;
  SPACING 0.10 ;
END Metal3
VIA M1_M2 DEFAULT
  LAYER Metal1 ;
    RECT -0.05 -0.05 0.05 0.05 ;
  LAYER Via12 ;
    RECT -0.04 -0.04 0.04 0.04 ;
  LAYER Metal2 ;
    RECT -0.05 -0.05 0.05 0.05 ;
END M1_M2
VIA M2_M3 DEFAULT
  LAYER Metal2 ;
    RECT -0.05 -0.05 0.05 0.05 ;
  LAYER Via23 ;
    RECT -0.04 -0.04 0.04 0.04 ;
  LAYER Metal3 ;
    RECT -0.05 -0.05 0.05 0.05 ;
END M2_M3
SITE CoreSite
  CLASS CORE ;
  SYMMETRY Y ;
  SIZE 0.2 BY 1.0 ;
END CoreSite
END LIBRARY
)lef";

constexpr llvm::StringLiteral kCellLef = R"lef(VERSION 5.8 ;
BUSBITCHARS "[]" ;
DIVIDERCHAR "/" ;
MACRO DFF_X1
  CLASS CORE ;
  ORIGIN 0 0 ;
  SIZE 0.8 BY 1.0 ;
  SYMMETRY X Y ;
  SITE CoreSite ;
  PIN D
    DIRECTION INPUT ;
    USE SIGNAL ;
    PORT
      LAYER Metal2 ;
      RECT 0.02 0.20 0.12 0.32 ;
    END
  END D
  PIN CLK
    DIRECTION INPUT ;
    USE CLOCK ;
    PORT
      LAYER Metal2 ;
      RECT 0.02 0.68 0.12 0.80 ;
    END
  END CLK
  PIN Q
    DIRECTION OUTPUT ;
    USE SIGNAL ;
    PORT
      LAYER Metal2 ;
      RECT 0.68 0.44 0.78 0.56 ;
    END
  END Q
END DFF_X1
MACRO BUF_X1
  CLASS CORE ;
  ORIGIN 0 0 ;
  SIZE 0.4 BY 1.0 ;
  SYMMETRY X Y ;
  SITE CoreSite ;
  PIN A
    DIRECTION INPUT ;
    USE SIGNAL ;
    PORT
      LAYER Metal2 ;
      RECT 0.02 0.40 0.12 0.60 ;
    END
  END A
  PIN Y
    DIRECTION OUTPUT ;
    USE SIGNAL ;
    PORT
      LAYER Metal2 ;
      RECT 0.28 0.40 0.38 0.60 ;
    END
  END Y
END BUF_X1
END LIBRARY
)lef";

constexpr llvm::StringLiteral kLiberty = R"lib(library (synthetic) {
  delay_model : table_lookup;
  time_unit : "1ns";
  voltage_unit : "1V";
  current_unit : "1mA";
  leakage_power_unit : "1nW";
  capacitive_load_unit (1, pf);
  cell (BUF_X1) {
    area : 0.4;
    cell_leakage_power : 0.01;
    pin (A) {
      direction : input;
      capacitance : 0.001;
    }
    pin (Y) {
      direction : output;
      function : "A";
    }
  }
  cell (DFF_X1) {
    area : 0.8;
    cell_leakage_power : 0.02;
    ff (IQ, IQN) {
      clocked_on : "CLK";
      next_state : "D";
    }
    pin (D) {
      direction : input;
      capacitance : 0.001;
    }
    pin (CLK) {
      direction : input;
      clock : true;
      capacitance : 0.001;
    }
    pin (Q) {
      direction : output;
      function : "IQ";
    }
  }
}
)lib";

constexpr llvm::StringLiteral kNetlist = R"verilog(module top(clk, d, q);
  input clk;
  input d;
  output q;
  DFF_X1 u0 (.CLK(clk), .D(d), .Q(q));
endmodule
)verilog";

constexpr llvm::StringLiteral kConstraints =
    "create_clock -name core_clock -period 2 [get_ports clk]\n";

constexpr llvm::StringLiteral kBlackBoxContract =
    "DFF_X1 input CLK input D output Q\n";

} // namespace

OpenRoadTechnologyFixture syntheticOpenRoadTechnologyFixture() {
  return {kTechnologyLef.str(),
          kCellLef.str(),
          kLiberty.str(),
          kNetlist.str(),
          kConstraints.str(),
          kBlackBoxContract.str(),
          {"DFF_X1"},
          OpenRoadPlacementParameters{{0, 0, 100000, 100000},
                                      {10000, 10000, 90000, 90000},
                                      "CoreSite",
                                      "Metal2",
                                      "Metal3",
                                      550000}};
}

llvm::Expected<OpenRoadTechnologyFixture>
loadSaed32OpenRoadTechnologyFixture(const std::filesystem::path &technologyLef,
                                    const std::filesystem::path &cellLef,
                                    const std::filesystem::path &liberty) {
  auto technology = readText(technologyLef);
  if (!technology)
    return technology.takeError();
  auto cells = readText(cellLef);
  if (!cells)
    return cells.takeError();
  auto timing = readText(liberty);
  if (!timing)
    return timing.takeError();
  return OpenRoadTechnologyFixture{
      std::move(*technology),
      std::move(*cells),
      std::move(*timing),
      R"verilog(module top(clk, d, q);
  input clk;
  input d;
  output q;
  DFFX1_RVT u0 (.CLK(clk), .D(d), .Q(q));
endmodule
)verilog",
      "create_clock -name core_clock -period 2 [get_ports clk]\n",
      "DFFX1_RVT input CLK input D output Q\n",
      {"DFFX1_RVT"},
      OpenRoadPlacementParameters{{0, 0, 100000, 100000},
                                  {10000, 10000, 90000, 90000},
                                  "unit",
                                  "M3",
                                  "M2",
                                  550000}};
}

llvm::Expected<OpenRoadTechnologyFixture>
loadGpdk045OpenRoadTechnologyFixture(const std::filesystem::path &technologyLef,
                                     const std::filesystem::path &cellLef,
                                     const std::filesystem::path &liberty) {
  auto technology = readText(technologyLef);
  if (!technology)
    return technology.takeError();
  auto cells = readText(cellLef);
  if (!cells)
    return cells.takeError();
  auto timing = readText(liberty);
  if (!timing)
    return timing.takeError();
  return OpenRoadTechnologyFixture{
      std::move(*technology),
      std::move(*cells),
      std::move(*timing),
      R"verilog(module top(clk, d, q);
  input clk;
  input d;
  output q;
  wire launched;
  wire inverted;
  DFFX1 launch (.CK(clk), .D(d), .Q(launched));
  INVX1 combinational (.A(launched), .Y(inverted));
  DFFX1 capture (.CK(clk), .D(inverted), .Q(q));
endmodule
)verilog",
      "create_clock -name core_clock -period 2 [get_ports clk]\n",
      "DFFX1 input CK input D output Q output QN\n"
      "INVX1 input A output Y\n",
      {"DFFX1", "INVX1"},
      OpenRoadPlacementParameters{{0, 0, 100000, 100000},
                                  {10000, 10000, 90000, 90000},
                                  "CoreSite",
                                  "Metal3",
                                  "Metal2",
                                  550000}};
}

llvm::Expected<std::string> readText(const std::filesystem::path &path) {
  auto buffer = llvm::MemoryBuffer::getFile(path.string());
  if (!buffer)
    return failure("cannot read " + path.string());
  return (*buffer)->getBuffer().str();
}

llvm::Error writeText(const std::filesystem::path &path,
                      llvm::StringRef contents, bool executable) {
  std::error_code error;
  std::filesystem::create_directories(path.parent_path(), error);
  if (error)
    return failure("cannot create directory for " + path.string());
  llvm::raw_fd_ostream output(path.string(), error);
  if (error)
    return failure("cannot write " + path.string() + ": " + error.message());
  output << contents;
  output.close();
  if (output.has_error())
    return failure("cannot finish writing " + path.string());
  if (!executable)
    return llvm::Error::success();
  std::filesystem::permissions(path,
                               std::filesystem::perms::owner_read |
                                   std::filesystem::perms::owner_write |
                                   std::filesystem::perms::owner_exec |
                                   std::filesystem::perms::group_read |
                                   std::filesystem::perms::group_exec,
                               std::filesystem::perm_options::replace, error);
  if (error)
    return failure("cannot make fixture executable: " + error.message());
  return llvm::Error::success();
}

ExternalFileFingerprint contentFingerprint(llvm::StringRef contents) {
  return llvm::cantFail(
      ExternalFileFingerprint::fromBytes(llvm::SHA256::hash(bytes(contents))));
}

llvm::Expected<OpenRoadGateFixture> makeOpenRoadGateFixture(
    const std::filesystem::path &root, const ArtifactStore &artifacts,
    const BlobStore &blobs, llvm::StringRef providerBuild,
    const OpenRoadTechnologyFixture &technology, llvm::StringRef designIdentity,
    std::uint32_t designPortBitWidth) {
  const std::filesystem::path external = root / "external";
  const std::filesystem::path technologyPath = external / "technology.lef";
  const std::filesystem::path cellsPath = external / "cells.lef";
  const std::filesystem::path libertyPath = external / "cells.lib";
  if (llvm::Error error = writeText(technologyPath, technology.technologyLef))
    return std::move(error);
  if (llvm::Error error = writeText(cellsPath, technology.cellLef))
    return std::move(error);
  if (llvm::Error error = writeText(libertyPath, technology.liberty))
    return std::move(error);

  adg::DesignBuilder design(artifacts);
  std::vector<adg::PortType> designPorts;
  if (designPortBitWidth != 0) {
    auto port = adg::PortType::bits(designPortBitWidth);
    if (!port)
      return port.takeError();
    designPorts.push_back(*port);
  }
  auto spatial =
      design.createSpatialCore(designIdentity, designPorts, designPorts);
  if (!spatial)
    return spatial.takeError();
  if (designPorts.empty()) {
    if (llvm::Error error = spatial->close({}))
      return std::move(error);
  } else {
    auto input = spatial->input(0);
    if (!input)
      return input.takeError();
    if (llvm::Error error = spatial->close({*input}))
      return std::move(error);
  }
  auto moduleDesign = std::move(design).finalize();
  if (!moduleDesign)
    return moduleDesign.takeError();
  if (moduleDesign->roots().size() != 1)
    return failure("fixture did not finalize one Fabric Module");
  auto module = fabric::importEntireFabricRoot(
      moduleDesign->roots().front().reference(), artifacts);
  if (!module)
    return module.takeError();
  auto system = hardware::test::makeSingleSpatialCoreSystem(*module, artifacts);
  if (!system)
    return system.takeError();
  auto subject = hardware::test::requireSingleSpatialCoreOccurrence(*system);
  if (!subject)
    return subject.takeError();
  auto abi = hardware::finalizeConfigurationABI(
      hardware::ConfigurationABIDraft{system->reference(), {}, {}}, artifacts);
  if (!abi)
    return abi.takeError();
  auto platform = platform::finalizeImplementationPlatform(
      platform::ImplementationPlatformDraft{
          platform::AsicTarget{"openroad-fixture", "2026.08"}, {"typical"}},
      artifacts);
  if (!platform)
    return platform.takeError();

  auto netlist = blobs.put(bytes(technology.netlist));
  if (!netlist)
    return netlist.takeError();
  auto constraints = blobs.put(bytes(technology.constraints));
  if (!constraints)
    return constraints.takeError();
  auto contract = blobs.put(bytes(technology.blackBoxContract));
  if (!contract)
    return contract.takeError();
  auto format = hardware::RepresentationFormatDescriptorRef::get(
      hardware::RepresentationFormatKind::StructuralVerilogGateNetlist);
  if (!format)
    return format.takeError();
  auto representation = hardware::createImplementationRepresentationRoot(
      hardware::RepresentationRootVariant::GateNetlist, std::nullopt, *format,
      {hardware::RepresentationObjectKind::Module, "top"},
      {{hardware::PayloadRole::Netlist, "netlist/top.v", *netlist},
       {hardware::PayloadRole::GenerationConstraint, "constraints/top.sdc",
        *constraints},
       {hardware::PayloadRole::BlackBoxContract,
        "contracts/yosys-standard-cells.txt", *contract}});
  if (!representation)
    return representation.takeError();
  auto contracts = makeKnownAsicStandardCellContractCatalog();
  if (!contracts)
    return contracts.takeError();
  const ExternalFileFingerprint libertyFingerprint =
      contentFingerprint(technology.liberty);
  std::vector<hardware::RepresentationLocator> unresolvedCells;
  unresolvedCells.reserve(technology.unresolvedCellModules.size());
  for (const std::string &module : technology.unresolvedCellModules)
    unresolvedCells.push_back(
        {hardware::RepresentationObjectKind::Module, module});
  auto gate = hardware::finalizeHardwareImplementation(
      hardware::HardwareImplementationDraft{
          system->reference(),
          *subject,
          abi->reference(),
          std::move(*representation),
          platform->reference(),
          {},
          {},
          {},
          {{openSourceYosysStandardCellContractRef.str(),
            {{asicStandardCellLibertyInputSlot.str(),
              hardware::ExplicitFileDependency{libertyFingerprint}}},
            {},
            std::move(unresolvedCells),
            hardware::ImplementationPayloadKey{
                hardware::PayloadRole::BlackBoxContract,
                "contracts/yosys-standard-cells.txt"}}}},
      *contracts, artifacts, blobs);
  if (!gate)
    return gate.takeError();

  OpenRoadPlacedConfig config{
      providerBuild.str(),
      {platform->reference().artifact, platform::TechnologyCornerId(0)},
      technology.placement,
      {{OpenRoadExternalFileKind::TechnologyLef, "technology",
        contentFingerprint(technology.technologyLef)},
       {OpenRoadExternalFileKind::CellLef, "cells",
        contentFingerprint(technology.cellLef)},
       {OpenRoadExternalFileKind::Liberty, "timing", libertyFingerprint}}};
  return OpenRoadGateFixture{std::move(*contracts),
                             std::move(*gate),
                             std::move(*platform),
                             std::move(config),
                             technologyPath,
                             cellsPath,
                             libertyPath};
}

llvm::Expected<OpenRoadRouteHarness>
makeOpenRoadRouteHarness(const std::filesystem::path &bundleRoot,
                         const OpenRoadGateFixture &fixture,
                         const external_tool::LocalToolConfig &localConfig) {
  auto configBytes = encodeOpenRoadPlacedConfig(fixture.config);
  if (!configBytes)
    return configBytes.takeError();
  auto configDigest = computeComponentViewDigest(
      openRoadPlacedConfigSchemaDescriptorBytes(), *configBytes);
  if (!configDigest)
    return configDigest.takeError();
  auto binding = dse::ResolvedCandidateGeneratorBinding::get(
      openRoadRoutedCandidateGeneratorDescriptor().reference(), *configBytes,
      *configDigest);
  if (!binding)
    return binding.takeError();
  return OpenRoadRouteHarness{
      {{dse::CandidateGeneratorInputSlotRef(0), {fixture.gate.reference()}}},
      std::move(*binding),
      external_tool::ExternalToolPreparationContext{localConfig,
                                                    bundleRoot.string()}};
}

external_tool::LocalToolConfig
makeOpenRoadLocalToolConfig(const OpenRoadGateFixture &fixture,
                            const std::filesystem::path &toolExecutable) {
  external_tool::LocalToolConfig local =
      external_tool::defaultLocalToolConfig();
  local.runtimePolicy = external_tool::RuntimePolicy::Host;
  local.tools["openroad"].binding.executable = toolExecutable.string();
  local.externalFiles = {{"technology", fixture.technologyLefPath.string()},
                         {"cells", fixture.cellLefPath.string()},
                         {"timing", fixture.libertyPath.string()}};
  return local;
}

OpenRoadResolvedExecution
makeOpenRoadResolvedExecution(llvm::StringRef executable,
                              llvm::StringRef version, bool moduleBound) {
  std::vector<std::string> modules;
  if (moduleBound)
    modules.push_back("openroad/2026.08.25-21512b0ab68c");
  external_tool::ExternalToolProviderDescriptor provider{
      external_tool::ToolProviderDescriptor{
          "openroad", {"openroad"}, {}, modules},
      external_tool::ToolVersionProbe{{"-version"},
                                      moduleBound ? "21512b0"
                                                  : version.str(),
                                      {0},
                                      std::nullopt},
      external_tool::ToolRuntimeCompatibility{}};
  external_tool::ResolvedToolBinding tool{
      "openroad",
      moduleBound ? external_tool::ToolBindingSource::Module
                  : external_tool::ToolBindingSource::Explicit,
      executable.str(),
      version.str(),
      modules,
      modules,
      moduleBound ? std::optional<std::string>("/etc/profile.d/modules.sh")
                  : std::nullopt,
      std::nullopt};
  external_tool::InvocationRuntimeBinding runtime;
  runtime.kind = external_tool::InvocationRuntimeKind::Host;
  return {std::move(provider), std::move(tool), std::move(runtime), {}};
}

llvm::Expected<std::filesystem::path>
writeAuthoredOpenRoadRouteTool(const std::filesystem::path &root,
                               AuthoredOpenRoadRouteBehavior behavior) {
  llvm::StringRef suffix;
  llvm::StringRef failure;
  llvm::StringRef missingOutput;
  switch (behavior) {
  case AuthoredOpenRoadRouteBehavior::Complete:
    suffix = "complete";
    break;
  case AuthoredOpenRoadRouteBehavior::ToolFailure:
    suffix = "tool-failure";
    failure = "exit 37\n";
    break;
  case AuthoredOpenRoadRouteBehavior::MissingOutput:
    suffix = "missing-output";
    missingOutput = "rm outputs/routed.def\n";
    break;
  }
  const std::filesystem::path tool =
      root / ("authored-openroad-route-" + suffix).str();
  const std::string body = R"sh(#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-version" || "${1:-}" == "--version" ]]; then
  printf '%s\n' 'OpenROAD synthetic 21512b0ab68c'
  exit 0
fi
if [[ "$#" -ne 7 || "$1" != "-no_init" || "$2" != "-no_splash" ||
      "$3" != "-no_settings" || "$4" != "-threads" || "$5" != "1" ||
      "$6" != "-exit" || "$7" != "drivers/openroad-routed.tcl" ]]; then
  exit 64
fi
grep -F 'clock_tree_synthesis -repair_clock_nets' drivers/openroad-routed.tcl >/dev/null
grep -F 'detailed_route -or_seed 1' drivers/openroad-routed.tcl >/dev/null
grep -F 'module top' inputs/netlist/0000.v >/dev/null
grep -F 'create_clock' inputs/constraints/0000.sdc >/dev/null
mkdir -p outputs work
)sh" + failure.str() + R"sh(cp inputs/netlist/0000.v outputs/routed.v
cat > outputs/routed.def <<'EOF'
VERSION 5.8 ;
DIVIDERCHAR "/" ;
BUSBITCHARS "[]" ;
DESIGN top ;
UNITS DISTANCE MICRONS 1000 ;
DIEAREA ( 0 0 ) ( 100000 100000 ) ;
COMPONENTS 1 ;
- u0 DFF_X1 + PLACED ( 20000 20000 ) N ;
END COMPONENTS
PINS 3 ;
- clk + NET clk + DIRECTION INPUT + USE CLOCK
  + LAYER Metal2 ( -50 -50 ) ( 50 50 ) + FIXED ( 10000 10000 ) N ;
- d + NET d + DIRECTION INPUT + USE SIGNAL
  + LAYER Metal2 ( -50 -50 ) ( 50 50 ) + FIXED ( 10000 30000 ) N ;
- q + NET q + DIRECTION OUTPUT + USE SIGNAL
  + LAYER Metal2 ( -50 -50 ) ( 50 50 ) + FIXED ( 90000 50000 ) N ;
END PINS
NETS 3 ;
- clk ( PIN clk ) ( u0 CLK ) + ROUTED Metal2 ( 10000 10000 ) ( 20000 20000 ) ;
- d ( PIN d ) ( u0 D ) + ROUTED Metal2 ( 10000 30000 ) ( 20000 20000 ) ;
- q ( PIN q ) ( u0 Q ) + ROUTED Metal2 ( 20000 20000 ) ( 90000 50000 ) ;
END NETS
END DESIGN
EOF
sed -n '/^set loom_result /,$p' drivers/openroad-routed.tcl > work/publish-result.tcl
tclsh work/publish-result.tcl
)sh" + missingOutput.str();
  if (llvm::Error error = writeText(tool, body, true))
    return std::move(error);
  return tool;
}

llvm::Expected<std::filesystem::path>
writeAuthoredOpenRoadStaticFpaTool(const std::filesystem::path &root,
                                   AuthoredOpenRoadStaticFpaBehavior behavior) {
  llvm::StringRef suffix;
  llvm::StringRef failure;
  llvm::StringRef malformedResult;
  switch (behavior) {
  case AuthoredOpenRoadStaticFpaBehavior::Complete:
    suffix = "complete";
    break;
  case AuthoredOpenRoadStaticFpaBehavior::ToolFailure:
    suffix = "tool-failure";
    failure = "exit 41\n";
    break;
  case AuthoredOpenRoadStaticFpaBehavior::MalformedResult:
    suffix = "malformed-result";
    malformedResult = "sed -i '0,/\"unit\":\"watt\"/s//\"unit\":\"volt\"/' "
                      "outputs/openroad-static-fpa-result.json\n";
    break;
  }
  const std::filesystem::path tool =
      root / ("authored-openroad-fpa-" + suffix).str();
  const std::string body = R"sh(#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-version" || "${1:-}" == "--version" ]]; then
  printf '%s\n' 'OpenROAD synthetic 21512b0ab68c'
  exit 0
fi
if [[ "$#" -ne 7 || "$1" != "-no_init" || "$2" != "-no_splash" ||
      "$3" != "-no_settings" || "$4" != "-threads" || "$5" != "1" ||
      "$6" != "-exit" || "$7" != "drivers/openroad-static-fpa.tcl" ]]; then
  exit 64
fi
grep -F 'read_def "inputs/database/routed.def"' drivers/openroad-static-fpa.tcl >/dev/null
grep -F 'extract_parasitics -version 2.0 -lef_rc' drivers/openroad-static-fpa.tcl >/dev/null
grep -F 'OpenRCX produced no parasitic segments' drivers/openroad-static-fpa.tcl >/dev/null
grep -E 'sta::find_clk_min_period|sta::design_power|rsz::design_area' drivers/openroad-static-fpa.tcl >/dev/null
grep -F 'DESIGN top' inputs/database/routed.def >/dev/null
grep -F 'module top' inputs/netlist/0.v >/dev/null
grep -F 'create_clock' inputs/constraints/0.sdc >/dev/null
mkdir -p work outputs
)sh" + failure.str() + R"sh({
  printf '%s\n' 'schema=loom.openroad_static_fpa_raw_report'
  printf '%s\n' 'version=1.0'
  printf '%s\n' 'top=top'
  grep -F 'limiting_clock_frequency_hz=%.12e' drivers/openroad-static-fpa.tcl >/dev/null && printf '%s\n' 'limiting_clock_frequency_hz=5.000000000000e+08' || true
  grep -F 'total_area_square_meters=%.12e' drivers/openroad-static-fpa.tcl >/dev/null && printf '%s\n' 'total_area_square_meters=1.200000000000e-10' || true
  grep -F 'dynamic_power_watts=%.12e' drivers/openroad-static-fpa.tcl >/dev/null && printf '%s\n' 'dynamic_power_watts=3.400000000000e-03' || true
  grep -F 'leakage_power_watts=%.12e' drivers/openroad-static-fpa.tcl >/dev/null && printf '%s\n' 'leakage_power_watts=5.600000000000e-04' || true
} > work/openroad-static-fpa-raw.txt
tclsh drivers/openroad-static-fpa-publish.tcl
)sh" + malformedResult.str();
  if (llvm::Error error = writeText(tool, body, true))
    return std::move(error);
  return tool;
}

llvm::Expected<hardware::FinalizedHardwareImplementation>
runOpenRoadRouteFixture(const OpenRoadGateFixture &fixture,
                        OpenRoadRouteHarness &harness,
                        const OpenRoadResolvedExecution &execution,
                        const ArtifactStore &artifacts,
                        const BlobStore &blobs) {
  auto prepared = prepareOpenRoadRoutedInvocation(
      harness.inputs, harness.binding, fixture.contracts, artifacts, blobs,
      execution, harness.context);
  if (!prepared)
    return prepared.takeError();
  auto exitCode = external_tool::executeExternalToolInvocationBundle(*prepared);
  if (!exitCode)
    return exitCode.takeError();
  if (*exitCode != 0)
    return failure("route fixture exited with status " +
                   std::to_string(*exitCode));
  auto result =
      importOpenRoadRoutedInvocation(harness.inputs, harness.binding, *prepared,
                                     fixture.contracts, artifacts, blobs);
  if (!result)
    return result.takeError();
  const auto *completed =
      std::get_if<dse::CompletedCandidateGeneratorResult>(&result->outcome);
  if (!completed || completed->outputBindings.size() != 1 ||
      completed->outputBindings.front().artifacts.size() != 1)
    return failure("route fixture did not publish one implementation");
  return hardware::importHardwareImplementation(
      completed->outputBindings.front().artifacts.front(), fixture.contracts,
      artifacts, blobs);
}

} // namespace loom::eda::open_source::test
