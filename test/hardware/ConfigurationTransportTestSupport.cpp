#include "ConfigurationTransportTestSupport.h"

#include "Fabric/Identity/FabricSemanticFieldRelation.h"

#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

namespace loom::hardware::test {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

std::uint32_t imageWord(llvm::ArrayRef<std::uint8_t> image,
                        std::uint64_t word) {
  std::uint32_t result = 0;
  for (unsigned byte = 0; byte != 4; ++byte) {
    const std::uint64_t index = word * 4 + byte;
    if (index < image.size())
      result |= std::uint32_t(image[static_cast<std::size_t>(index)])
                << (byte * 8);
  }
  return result;
}

std::uint8_t imageStrobe(llvm::ArrayRef<std::uint8_t> image,
                         std::uint64_t word) {
  const std::uint64_t firstByte = word * 4;
  if (firstByte >= image.size())
    return 0;
  const unsigned count = static_cast<unsigned>(
      std::min<std::uint64_t>(4, image.size() - firstByte));
  return static_cast<std::uint8_t>((1U << count) - 1U);
}

} // namespace

llvm::Expected<PortableConfigurationTarget> derivePortableConfigurationTarget(
    const FinalizedConfigurationABI &configurationAbi,
    fabric::SpatialCoreOccurrenceRef spatialCore, ProgrammingUnitId unitId) {
  auto layout = rtl::derivePortableConfigurationTransportLayout(
      configurationAbi, spatialCore);
  if (!layout)
    return layout.takeError();
  const rtl::ConfigurationTransportUnitLayout *unit = layout->find(unitId);
  if (!unit)
    return invalid("programming unit is absent from the local transport");
  return PortableConfigurationTarget{unitId,
                                     unit->payloadBitCount,
                                     unit->payloadByteCount,
                                     unit->payloadWordCount,
                                     unit->baseAddress,
                                     unit->commitAddress,
                                     unit->statusAddress};
}

llvm::Expected<PortableConfigurationValue>
deriveSpatialSingleTemplateFuActivation(
    const fabric::FabricArtifactView &fabric,
    const FinalizedConfigurationABI &configurationAbi,
    fabric::SpatialCoreOccurrenceRef spatialCore,
    fabric::FabricFuOccurrenceRef fu) {
  const auto definition = fabric.fuTemplateOf(fu);
  if (!definition)
    return invalid("FU occurrence has no template definition");
  const auto templates = fabric.fuCapabilityTemplates(*definition);
  if (templates.size() != 1)
    return invalid("test FU does not have exactly one capability template");

  const fabric::FabricSemanticConfigFieldRef semanticField{
      fabric::FabricConfigurationOwnerRef(
          fabric::FabricInventoryOwnerRef::of(fu)),
      0};
  auto semantic = fabric::encodeFabricFuConfiguration(
      fabric, semanticField,
      fabric::FabricFuCapabilityTemplateRef{*definition, 0});
  if (!semantic)
    return semantic.takeError();
  auto target = fabric::FabricModulePhysicalTargetRef::create(semanticField);
  if (!target)
    return target.takeError();
  auto physical = fabric::FabricPhysicalConfigurationFieldRef::create(
      fabric::SpatialCoreInternalOccurrenceRef{spatialCore,
                                               std::move(*target)});
  if (!physical)
    return physical.takeError();
  auto slot = fabric::qualifyFabricConfigurationSlot(
      *physical, fabric::FabricStaticConfigurationResidency{});
  if (!slot)
    return slot.takeError();

  const ProgrammingUnit *owner = nullptr;
  for (const ProgrammingUnit &unit : configurationAbi.abi().programmingUnits())
    for (const ConfigurationFieldEncoding &field : unit.fields)
      if (field.slot == *slot) {
        if (owner)
          return invalid("FU topology field has duplicate programming owners");
        owner = &unit;
      }
  if (!owner)
    return invalid("FU topology field has no programming owner");
  return PortableConfigurationValue{
      owner->id, SemanticConfigurationValue{
                     std::move(*slot),
                     std::vector<std::uint8_t>(semantic->bytes().begin(),
                                               semantic->bytes().end())}};
}

std::string portableAxiLiteSignalDeclarations() {
  return R"sv(  logic [31:0] cfg_awaddr;
  logic        cfg_awvalid;
  logic        cfg_awready;
  logic [31:0] cfg_wdata;
  logic [3:0]  cfg_wstrb;
  logic        cfg_wvalid;
  logic        cfg_wready;
  logic [1:0]  cfg_bresp;
  logic        cfg_bvalid;
  logic        cfg_bready;
  logic [31:0] cfg_araddr;
  logic        cfg_arvalid;
  logic        cfg_arready;
  logic [31:0] cfg_rdata;
  logic [1:0]  cfg_rresp;
  logic        cfg_rvalid;
  logic        cfg_rready;
  logic [31:0] cfg_readback;
  logic [31:0] cfg_active_snapshot;
  logic [1:0]  cfg_read_response;
  integer      loom_verbose_level;
  initial begin
    if (!$value$plusargs("LOOM_VERBOSE_LEVEL=%d", loom_verbose_level))
      loom_verbose_level = 0;
  end
)sv";
}

std::string portableAxiLiteDriverTasks() {
  return R"sv(
  task automatic cfg_finish_write(
      input logic [1:0] expected_response);
    begin
      #1;
      cfg_awvalid = 0;
      cfg_wvalid = 0;
      if (cfg_bvalid !== 1'b1)
        $fatal(1, "AXI4-Lite write did not complete on the accepting edge");
      if (cfg_bresp !== expected_response)
        $fatal(1, "unexpected AXI4-Lite write response");
      @(negedge clock);
    end
  endtask

  task automatic cfg_write(
      input logic [31:0] address,
      input logic [31:0] data,
      input logic [3:0] strobe,
      input logic [1:0] expected_response);
    integer wait_cycles;
    begin
      if (loom_verbose_level >= 1)
        $display("[loom][config][write] address=%h data=%h strobe=%h",
                 address, data, strobe);
      cfg_awaddr = address;
      cfg_awvalid = 1;
      wait_cycles = 0;
      do begin
        @(posedge clock);
        wait_cycles = wait_cycles + 1;
        if (loom_verbose_level >= 3)
          $display("[loom][config][aw] cycle=%0d reset=%b valid=%b ready=%b",
                   wait_cycles, reset, cfg_awvalid, cfg_awready);
        if (wait_cycles == 64 && !cfg_awready)
          $fatal(1, "AXI4-Lite AW handshake timed out");
      end while (!cfg_awready);
      if (loom_verbose_level >= 2)
        $display("[loom][config][aw] accepted address=%h", address);
      #1;
      cfg_awvalid = 0;
      cfg_awaddr = ~address;

      cfg_wdata = data;
      cfg_wstrb = strobe;
      cfg_wvalid = 1;
      wait_cycles = 0;
      do begin
        @(posedge clock);
        wait_cycles = wait_cycles + 1;
        if (loom_verbose_level >= 3)
          $display("[loom][config][w] cycle=%0d reset=%b valid=%b ready=%b",
                   wait_cycles, reset, cfg_wvalid, cfg_wready);
        if (wait_cycles == 64 && !cfg_wready)
          $fatal(1, "AXI4-Lite W handshake timed out");
      end while (!cfg_wready);
      if (loom_verbose_level >= 2)
        $display("[loom][config][w] accepted data=%h strobe=%h", data,
                 strobe);
      cfg_finish_write(expected_response);
    end
  endtask

  task automatic cfg_write_together(
      input logic [31:0] address,
      input logic [31:0] data,
      input logic [3:0] strobe,
      input logic [1:0] expected_response);
    integer wait_cycles;
    logic address_accepted;
    logic data_accepted;
    begin
      cfg_awaddr = address;
      cfg_awvalid = 1;
      cfg_wdata = data;
      cfg_wstrb = strobe;
      cfg_wvalid = 1;
      address_accepted = 0;
      data_accepted = 0;
      wait_cycles = 0;
      do begin
        @(posedge clock);
        address_accepted = address_accepted | cfg_awready;
        data_accepted = data_accepted | cfg_wready;
        wait_cycles = wait_cycles + 1;
        if (wait_cycles == 64 &&
            (!address_accepted || !data_accepted))
          $fatal(1, "simultaneous AXI4-Lite write handshake timed out");
      end while (!address_accepted || !data_accepted);
      cfg_finish_write(expected_response);
    end
  endtask

  task automatic cfg_write_data_first(
      input logic [31:0] address,
      input logic [31:0] data,
      input logic [3:0] strobe,
      input logic [1:0] expected_response);
    integer wait_cycles;
    begin
      cfg_wdata = data;
      cfg_wstrb = strobe;
      cfg_wvalid = 1;
      wait_cycles = 0;
      do begin
        @(posedge clock);
        wait_cycles = wait_cycles + 1;
        if (wait_cycles == 64 && !cfg_wready)
          $fatal(1, "AXI4-Lite W handshake timed out");
      end while (!cfg_wready);
      #1;
      cfg_wvalid = 0;
      cfg_wdata = ~data;
      cfg_wstrb = ~strobe;

      cfg_awaddr = address;
      cfg_awvalid = 1;
      wait_cycles = 0;
      do begin
        @(posedge clock);
        wait_cycles = wait_cycles + 1;
        if (wait_cycles == 64 && !cfg_awready)
          $fatal(1, "AXI4-Lite AW handshake timed out");
      end while (!cfg_awready);
      cfg_finish_write(expected_response);
    end
  endtask

  task automatic cfg_read(
      input logic [31:0] address,
      output logic [31:0] data,
      output logic [1:0] response);
    integer wait_cycles;
    begin
      if (loom_verbose_level >= 1)
        $display("[loom][config][read] address=%h", address);
      cfg_araddr = address;
      cfg_arvalid = 1;
      wait_cycles = 0;
      do begin
        @(posedge clock);
        wait_cycles = wait_cycles + 1;
        if (loom_verbose_level >= 3)
          $display("[loom][config][ar] cycle=%0d reset=%b valid=%b ready=%b",
                   wait_cycles, reset, cfg_arvalid, cfg_arready);
        if (wait_cycles == 64 && !cfg_arready)
          $fatal(1, "AXI4-Lite AR handshake timed out");
      end while (!cfg_arready);
      if (loom_verbose_level >= 2)
        $display("[loom][config][ar] accepted address=%h", address);
      #1 cfg_arvalid = 0;

      wait_cycles = 0;
      do begin
        @(negedge clock);
        wait_cycles = wait_cycles + 1;
        if (loom_verbose_level >= 3)
          $display("[loom][config][r] cycle=%0d valid=%b response=%b data=%h",
                   wait_cycles, cfg_rvalid, cfg_rresp, cfg_rdata);
        if (wait_cycles == 64 && !cfg_rvalid)
          $fatal(1, "AXI4-Lite R response timed out");
      end while (!cfg_rvalid);
      data = cfg_rdata;
      response = cfg_rresp;
    end
  endtask
)sv";
}

std::string portableCycleWatchdog(std::uint64_t cycleLimit) {
  std::string result;
  llvm::raw_string_ostream output(result);
  output << "\n  initial begin\n"
         << "    repeat (" << cycleLimit << ") @(posedge clock);\n"
         << "    $fatal(1, \"RTL testbench cycle limit exceeded\");\n"
         << "  end\n";
  return result;
}

std::string portableAxiLiteInitialization() {
  return R"sv(    cfg_awaddr = 0;
    cfg_awvalid = 0;
    cfg_wdata = 0;
    cfg_wstrb = 0;
    cfg_wvalid = 0;
    cfg_bready = 1;
    cfg_araddr = 0;
    cfg_arvalid = 0;
    cfg_rready = 1;
)sv";
}

llvm::Expected<std::string>
portableAxiLiteProgramAndVerify(const PortableConfigurationTarget &target,
                                llvm::ArrayRef<std::uint8_t> image,
                                llvm::StringRef indentation) {
  if (image.size() != target.payloadByteCount)
    return invalid("configuration image byte count disagrees with its target");
  if (target.payloadBitCount == 0 || target.payloadWordCount == 0)
    return invalid("configuration target has an empty payload");
  const unsigned lastBits = static_cast<unsigned>(target.payloadBitCount % 8);
  if (lastBits != 0 &&
      (image.back() & static_cast<std::uint8_t>(0xffU << lastBits)) != 0)
    return invalid("configuration image sets an ABI-unused high bit");

  std::string result;
  llvm::raw_string_ostream output(result);
  if (const unsigned usedLastWordBits =
          static_cast<unsigned>(target.payloadBitCount % 32);
      usedLastWordBits != 0) {
    const std::uint32_t address =
        target.baseAddress +
        static_cast<std::uint32_t>((target.payloadWordCount - 1) * 4);
    const std::uint32_t invalidBit = std::uint32_t{1} << usedLastWordBits;
    const std::uint32_t invalidStrobe = std::uint32_t{1}
                                        << (usedLastWordBits / 8);
    output << indentation << "cfg_write(32'h"
           << llvm::format_hex_no_prefix(address, 8) << ", 32'h"
           << llvm::format_hex_no_prefix(invalidBit, 8) << ", 4'h"
           << llvm::format_hex_no_prefix(invalidStrobe, 1) << ", 2'b10);\n";
  }
  if (target.payloadByteCount > 1) {
    output << indentation << "cfg_read(32'h"
           << llvm::format_hex_no_prefix(target.baseAddress, 8)
           << ", cfg_active_snapshot, cfg_read_response);\n"
           << indentation
           << "if (cfg_read_response !== 2'b00) "
              "$fatal(1, \"active configuration snapshot failed\");\n";
    const std::uint64_t omittedByte = target.payloadByteCount - 1;
    for (std::uint64_t word = 0; word != target.payloadWordCount; ++word) {
      std::uint8_t strobe = imageStrobe(image, word);
      if (omittedByte / 4 == word)
        strobe &= static_cast<std::uint8_t>(~(1U << (omittedByte % 4)));
      if (strobe == 0)
        continue;
      const std::uint32_t address =
          target.baseAddress + static_cast<std::uint32_t>(word * 4);
      output << indentation << "cfg_write(32'h"
             << llvm::format_hex_no_prefix(address, 8) << ", 32'h"
             << llvm::format_hex_no_prefix(imageWord(image, word), 8) << ", 4'h"
             << llvm::format_hex_no_prefix(strobe, 1) << ", 2'b00);\n";
    }
    output
        << indentation << "cfg_write(32'h"
        << llvm::format_hex_no_prefix(target.baseAddress, 8) << ", 32'h"
        << llvm::format_hex_no_prefix(imageWord(image, 0), 8)
        << ", 4'h1, 2'b00);\n"
        << indentation << "cfg_write(32'h"
        << llvm::format_hex_no_prefix(target.commitAddress, 8)
        << ", 32'h00000001, 4'h1, 2'b10);\n"
        << indentation << "cfg_read(32'h"
        << llvm::format_hex_no_prefix(target.baseAddress, 8)
        << ", cfg_readback, cfg_read_response);\n"
        << indentation
        << "if (cfg_read_response !== 2'b00 || "
           "cfg_readback !== cfg_active_snapshot) "
           "$fatal(1, \"incomplete commit changed active configuration\");\n";
  }
  for (std::uint64_t word = 0; word != target.payloadWordCount; ++word) {
    const std::uint32_t address =
        target.baseAddress + static_cast<std::uint32_t>(word * 4);
    output << indentation << "cfg_write(32'h"
           << llvm::format_hex_no_prefix(address, 8) << ", 32'h"
           << llvm::format_hex_no_prefix(imageWord(image, word), 8) << ", 4'h"
           << llvm::format_hex_no_prefix(imageStrobe(image, word), 1)
           << ", 2'b00);\n";
  }
  output << indentation << "cfg_write_data_first(32'h"
         << llvm::format_hex_no_prefix(target.commitAddress, 8)
         << ", 32'h00000001, 4'h1, 2'b00);\n";
  for (std::uint64_t word = 0; word != target.payloadWordCount; ++word) {
    const std::uint32_t address =
        target.baseAddress + static_cast<std::uint32_t>(word * 4);
    output << indentation << "cfg_read(32'h"
           << llvm::format_hex_no_prefix(address, 8)
           << ", cfg_readback, cfg_read_response);\n"
           << indentation
           << "if (cfg_read_response !== 2'b00 || cfg_readback !== 32'h"
           << llvm::format_hex_no_prefix(imageWord(image, word), 8)
           << ") $fatal(1, \"active configuration readback mismatch\");\n";
  }
  output << indentation << "cfg_read(32'h"
         << llvm::format_hex_no_prefix(target.statusAddress, 8)
         << ", cfg_readback, cfg_read_response);\n"
         << indentation
         << "if (cfg_read_response !== 2'b00 || cfg_readback !== 32'h00000000) "
            "$fatal(1, \"configuration status did not clear after commit\");\n";
  return result;
}

} // namespace loom::hardware::test
