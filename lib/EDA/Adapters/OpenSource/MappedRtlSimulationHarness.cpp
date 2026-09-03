#include "MappedRtlSimulationInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>

namespace loom::eda::open_source::detail {
namespace {

constexpr unsigned kBitsPerByte = 8;
constexpr unsigned kBitsPerHexDigit = 4;
constexpr unsigned kAxiResponseWidth = 2;
constexpr unsigned kConfigurationProgramWordWidth = 36;
constexpr unsigned kConfigurationHandshakeCycleLimit = 64;
constexpr unsigned kResetReleaseCycleCount = 4;
constexpr std::uint64_t kMinimumClockPeriodFs = 2;
/// The harness carries byte addresses as `longint unsigned`; that type is the
/// portable byte-address domain, so the two must agree.
static_assert(hardware::rtl::portableMemoryByteAddressWidthBits == 64);
static_assert(hardware::rtl::portableConfigurationDataWidth == 32);
static_assert(hardware::rtl::portableConfigurationByteCount == 4);

llvm::Error invalid(const llvm::Twine &detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "mapped_rtl_harness_invalid: " + detail);
}

std::string hexBits(const llvm::APInt &value) {
  llvm::SmallString<128> digits;
  value.toString(digits, 16, false);
  std::string result;
  const std::size_t required =
      (value.getBitWidth() + kBitsPerHexDigit - 1) / kBitsPerHexDigit;
  result.append(required - digits.size(), '0');
  result.append(digits.begin(), digits.end());
  return result;
}

std::uint32_t imageWord(llvm::ArrayRef<std::uint8_t> image,
                        std::uint64_t word) {
  std::uint32_t result = 0;
  for (unsigned byte = 0; byte != hardware::rtl::portableConfigurationByteCount;
       ++byte) {
    const std::uint64_t index =
        word * hardware::rtl::portableConfigurationByteCount + byte;
    if (index < image.size())
      result |= std::uint32_t(image[index]) << (byte * kBitsPerByte);
  }
  return result;
}

std::uint8_t imageStrobe(llvm::ArrayRef<std::uint8_t> image,
                         std::uint64_t word) {
  const std::uint64_t first =
      word * hardware::rtl::portableConfigurationByteCount;
  if (first >= image.size())
    return 0;
  const unsigned count = static_cast<unsigned>(std::min<std::uint64_t>(
      hardware::rtl::portableConfigurationByteCount, image.size() - first));
  return static_cast<std::uint8_t>((1U << count) - 1U);
}

llvm::Error validateProgram(const ConfigurationProgram &program) {
  if (program.portPrefix.empty() ||
      program.image.size() != program.layout.payloadByteCount ||
      program.layout.payloadBitCount == 0 ||
      program.layout.payloadWordCount == 0)
    return invalid("configuration program is incomplete");
  const unsigned lastBits =
      static_cast<unsigned>(program.layout.payloadBitCount % kBitsPerByte);
  if (lastBits != 0 && (program.image.back() &
                        static_cast<std::uint8_t>(0xffU << lastBits)) != 0)
    return invalid("configuration image sets an ABI-unused high bit");
  return llvm::Error::success();
}

void renderPortDeclaration(llvm::raw_ostream &output, const RtlPort &port) {
  output << "  logic";
  if (port.bitWidth > 1)
    output << " [" << port.bitWidth - 1 << ":0]";
  output << " " << port.name << ";\n";
}

void renderPhysicalTagAssertion(llvm::raw_ostream &output,
                                const TransportPort &port,
                                llvm::StringRef clock,
                                llvm::StringRef observationKind,
                                std::size_t ordinal) {
  if (!port.physicalTag)
    return;
  output << "  always_ff @(posedge " << clock << ")\n"
         << "    if (loom_resets_released && " << port.prefix << "_valid && "
         << port.prefix << "_ready && " << port.prefix
         << "_tag !== " << port.physicalTag->getBitWidth() << "'h"
         << hexBits(*port.physicalTag) << ") $fatal(1, \"mapped "
         << observationKind << " " << ordinal
         << " carried the wrong Physical Tag\");\n";
}

std::set<std::string> customDrivenPorts(const MappedRtlInvocationFacts &facts) {
  std::set<std::string> names;
  for (const ClockPort &clock : facts.clockPorts)
    names.insert(clock.name);
  for (const ResetPort &reset : facts.resetPorts)
    names.insert(reset.name);
  const auto addInput = [&](const InputTokenStream &input) {
    names.insert(input.port.prefix + "_valid");
    if (input.port.payloadBitWidth != 0)
      names.insert(input.port.prefix + "_data");
    if (input.port.physicalTag)
      names.insert(input.port.prefix + "_tag");
  };
  if (facts.startInput)
    addInput(*facts.startInput);
  for (const InputTokenStream &input : facts.valueInputs)
    addInput(input);
  for (const InputTokenStream &input : facts.streamInputs)
    addInput(input);
  for (const ConfigurationProgram &program : facts.configurationPrograms) {
    for (llvm::StringRef suffix :
         {"_awaddr", "_awvalid", "_wdata", "_wstrb", "_wvalid", "_bready",
          "_araddr", "_arvalid", "_rready"})
      names.insert(program.portPrefix + suffix.str());
  }
  for (const MemoryBoundaryPort &memory : facts.memoryBoundaryPorts) {
    names.insert(memory.prefix + "_request_ready");
    names.insert(memory.prefix + "_response_data");
    names.insert(memory.prefix + "_response_valid");
  }
  return names;
}

llvm::Expected<std::uint64_t>
memoryExtent(const MappedRtlInvocationFacts &facts) {
  std::uint64_t extent = 0;
  for (const RuntimeMemoryImage &image : facts.memoryImages) {
    if (image.initialBytes.empty() ||
        image.initialBytes.size() > std::numeric_limits<std::uint64_t>::max() -
                                        image.canonicalBaseAddress)
      return invalid("runtime memory image has an invalid extent");
    extent = std::max<std::uint64_t>(extent, image.canonicalBaseAddress +
                                                 image.initialBytes.size());
  }
  return extent;
}

void renderMemoryDeclarations(llvm::raw_ostream &output,
                              const MappedRtlInvocationFacts &facts,
                              std::uint64_t extent) {
  if (extent == 0)
    return;
  output << "  logic [7:0] loom_runtime_memory [0:" << extent - 1 << "];\n"
         << "  function automatic bit loom_memory_address_valid("
            "input longint unsigned address);\n"
         << "    begin\n"
         << "      loom_memory_address_valid = 0;\n";
  for (const RuntimeMemoryImage &image : facts.memoryImages)
    output << "      if (address >= " << image.canonicalBaseAddress
           << " && address < "
           << image.canonicalBaseAddress + image.initialBytes.size()
           << ") loom_memory_address_valid = 1;\n";
  output << "    end\n"
         << "  endfunction\n";
  for (const auto &[ordinal, port] :
       llvm::enumerate(facts.memoryBoundaryPorts)) {
    output << "  logic loom_memory_response_pending_" << ordinal << ";\n"
           << "  logic [" << port.dataBitWidth - 1
           << ":0] loom_memory_response_data_" << ordinal << ";\n"
           << "  logic [" << port.dataBitWidth - 1
           << ":0] loom_memory_response_next_" << ordinal << ";\n"
           << "  longint unsigned loom_memory_element_bytes_" << ordinal
           << ";\n"
           << "  longint unsigned loom_memory_lane_" << ordinal << ";\n"
           << "  longint unsigned loom_memory_byte_in_lane_" << ordinal << ";\n"
           << "  longint unsigned loom_memory_lane_address_" << ordinal << ";\n"
           << "  longint unsigned loom_memory_lane_offset_" << ordinal << ";\n"
           << "  logic [" << facts.addressArithmetic.calculationWidthBits - 1
           << ":0] loom_memory_wide_address_" << ordinal << ";\n"
           << "  longint unsigned loom_memory_byte_address_" << ordinal << ";\n"
           << "  longint unsigned loom_memory_root_base_" << ordinal << ";\n"
           << "  logic loom_memory_context_matched_" << ordinal << ";\n"
           << "  integer loom_memory_byte_ordinal_" << ordinal << ";\n";
  }
}

void renderMemoryInitialization(llvm::raw_ostream &output,
                                const MappedRtlInvocationFacts &facts,
                                std::uint64_t extent) {
  if (extent == 0)
    return;
  output << "    for (integer loom_memory_init = 0; loom_memory_init < "
         << extent
         << "; loom_memory_init = loom_memory_init + 1) "
            "loom_runtime_memory[loom_memory_init] = 0;\n";
  for (const RuntimeMemoryImage &image : facts.memoryImages)
    for (const auto &[ordinal, byte] : llvm::enumerate(image.initialBytes))
      output << "    loom_runtime_memory["
             << image.canonicalBaseAddress + ordinal << "] = 8'h"
             << llvm::format_hex_no_prefix(byte.value, 2, true) << ";\n";
}

void renderMemoryService(llvm::raw_ostream &output,
                         const MappedRtlInvocationFacts &facts) {
  for (const auto &[ordinal, port] :
       llvm::enumerate(facts.memoryBoundaryPorts)) {
    output << "  always_comb begin\n"
           << "    loom_memory_root_base_" << ordinal << " = 0;\n"
           << "    loom_memory_context_matched_" << ordinal << " = 0;\n"
           << "    case (" << port.prefix << "_request_context)\n";
    for (const MemoryBoundaryBinding &binding : port.bindings) {
      const RuntimeMemoryImage &root =
          facts.memoryImages[binding.rootObjectOrdinal];
      output << "      64'h"
             << llvm::format_hex_no_prefix(binding.requestContext, 16, true)
             << ": begin\n"
             << "        loom_memory_root_base_" << ordinal << " = "
             << root.canonicalBaseAddress + binding.rootByteOffset << ";\n"
             << "        loom_memory_context_matched_" << ordinal << " = 1;\n"
             << "      end\n";
    }
    output << "      default: begin end\n"
           << "    endcase\n"
           << "  end\n"
           << "  always_comb begin\n"
           << "    " << port.prefix
           << "_request_ready = !loom_memory_response_pending_" << ordinal
           << " || " << port.prefix << "_response_ready;\n"
           << "    " << port.prefix
           << "_response_valid = loom_memory_response_pending_" << ordinal
           << ";\n"
           << "    " << port.prefix
           << "_response_data = loom_memory_response_data_" << ordinal << ";\n"
           << "  end\n";
  }
  if (facts.memoryBoundaryPorts.empty())
    return;

  // The runtime memory is also written by the initialization block, and
  // SystemVerilog forbids a second procedural driver of an `always_ff`
  // variable, so the service is a general clocked process.
  output << "  always @(posedge " << facts.selectedClock << ") begin\n"
         << "    if (!loom_resets_released) begin\n";
  for (const auto &[ordinal, port] :
       llvm::enumerate(facts.memoryBoundaryPorts)) {
    (void)port;
    output << "      loom_memory_response_pending_" << ordinal << " <= 0;\n"
           << "      loom_memory_response_data_" << ordinal << " <= '0;\n";
  }
  output << "    end else begin\n";
  const std::string wideCast =
      std::to_string(facts.addressArithmetic.calculationWidthBits) + "'";
  for (const auto &[ordinal, port] :
       llvm::enumerate(facts.memoryBoundaryPorts)) {
    const std::uint64_t dataBytes = port.dataBitWidth / kBitsPerByte;
    output << "      if (loom_memory_response_pending_" << ordinal << " && "
           << port.prefix << "_response_ready) "
           << "loom_memory_response_pending_" << ordinal << " <= 0;\n"
           << "      if (" << port.prefix << "_request_valid && " << port.prefix
           << "_request_ready) begin\n"
           << "        loom_memory_response_next_" << ordinal << " = '0;\n"
           << "        if (!loom_memory_context_matched_" << ordinal
           << ") $fatal(1, \"unknown external memory request context\");\n"
           << "        if (" << port.prefix << "_request_element_width == 0 || "
           << port.prefix << "_request_element_width[2:0] != 0 || "
           << port.prefix << "_request_element_width > " << port.dataBitWidth
           << ") $fatal(1, \"unsupported external memory element width\");\n"
           << "        if (" << port.prefix
           << "_request_address_lane_width == 0 || " << port.prefix
           << "_request_address_lane_width > "
           << facts.addressArithmetic.byteAddressWidthBits
           << ") $fatal(1, "
              "\"unsupported external memory address width\");\n"
           << "        loom_memory_element_bytes_" << ordinal << " = "
           << port.prefix << "_request_element_width >> 3;\n"
           << "        if (" << port.prefix << "_request_lane_count == 0 || "
           << port.prefix << "_request_lane_count * "
           << "loom_memory_element_bytes_" << ordinal << " > " << dataBytes
           << ") $fatal(1, \"external memory request exceeds data carrier\");\n"
           << "        if (" << port.prefix << "_request_active_lanes_kind && "
           << port.prefix << "_request_lane_count > " << port.maskBitWidth
           << ") $fatal(1, \"external memory mask is too narrow\");\n"
           << "        if (" << port.prefix << "_request_access_form == 2 && "
           << port.prefix << "_request_lane_count * " << port.prefix
           << "_request_address_lane_width > " << port.addressBitWidth
           << ") $fatal(1, \"indexed external memory address is too wide\");\n"
           << "        for (loom_memory_byte_ordinal_" << ordinal << " = 0; "
           << "loom_memory_byte_ordinal_" << ordinal << " < " << dataBytes
           << "; loom_memory_byte_ordinal_" << ordinal
           << " = loom_memory_byte_ordinal_" << ordinal << " + 1) begin\n"
           << "          loom_memory_lane_" << ordinal << " = "
           << "loom_memory_byte_ordinal_" << ordinal
           << " / loom_memory_element_bytes_" << ordinal << ";\n"
           << "          loom_memory_byte_in_lane_" << ordinal << " = "
           << "loom_memory_byte_ordinal_" << ordinal
           << " % loom_memory_element_bytes_" << ordinal << ";\n"
           << "          if (loom_memory_lane_" << ordinal << " < "
           << port.prefix << "_request_lane_count && (!" << port.prefix
           << "_request_active_lanes_kind || " << port.prefix
           << "_request_mask[loom_memory_lane_" << ordinal << "])) begin\n"
           << "            loom_memory_lane_address_" << ordinal << " = "
           << port.prefix << "_request_address >> ((" << port.prefix
           << "_request_access_form == 2) ? (loom_memory_lane_" << ordinal
           << " * " << port.prefix << "_request_address_lane_width) : 0);\n"
           << "            if (" << port.prefix
           << "_request_address_lane_width < "
           << facts.addressArithmetic.byteAddressWidthBits
           << ") loom_memory_lane_address_" << ordinal
           << " = loom_memory_lane_address_" << ordinal << " & (("
           << facts.addressArithmetic.byteAddressWidthBits << "'h1 << "
           << port.prefix << "_request_address_lane_width) - 1);\n"
           << "            loom_memory_lane_offset_" << ordinal << " = ("
           << port.prefix << "_request_access_form == 2) ? "
           << "loom_memory_byte_in_lane_" << ordinal << " : "
           << "loom_memory_byte_ordinal_" << ordinal
           << ";\n"
           // The complete byte-address expression is evaluated in the
           // portable calculation width; a request whose exact address
           // leaves the byte-address domain is a typed failure, never an
           // alias of the wrapped low bits.
           << "            if (" << port.prefix
           << "_request_address_form == 0) loom_memory_wide_address_" << ordinal
           << " = " << wideCast << "(loom_memory_root_base_" << ordinal
           << ") + " << wideCast << "(" << port.prefix
           << "_request_base_address) + " << wideCast
           << "(loom_memory_lane_address_" << ordinal << ") * " << wideCast
           << "(loom_memory_element_bytes_" << ordinal << ") + " << wideCast
           << "(loom_memory_lane_offset_" << ordinal << ");\n"
           << "            else loom_memory_wide_address_" << ordinal << " = "
           << wideCast << "(loom_memory_lane_address_" << ordinal << ") + "
           << wideCast << "(loom_memory_lane_offset_" << ordinal << ");\n"
           << "            if (|loom_memory_wide_address_" << ordinal << "["
           << facts.addressArithmetic.calculationWidthBits - 1 << ':'
           << facts.addressArithmetic.byteAddressWidthBits
           << "]) $fatal(1, \"external memory address overflows the portable "
              "byte-address domain\");\n"
           << "            loom_memory_byte_address_" << ordinal
           << " = loom_memory_wide_address_" << ordinal << "["
           << facts.addressArithmetic.byteAddressWidthBits - 1 << ":0];\n"
           << "            if (!loom_memory_address_valid("
           << "loom_memory_byte_address_" << ordinal
           << ")) $fatal(1, \"external memory address is out of range\");\n"
           << "            if (" << port.prefix
           << "_request_kind == 0) loom_memory_response_next_" << ordinal
           << "[loom_memory_byte_ordinal_" << ordinal << " * 8 +: 8] = "
           << "loom_runtime_memory[loom_memory_byte_address_" << ordinal
           << "];\n"
           << "            else loom_runtime_memory[loom_memory_byte_address_"
           << ordinal << "] <= " << port.prefix
           << "_request_data[loom_memory_byte_ordinal_" << ordinal
           << " * 8 +: 8];\n"
           << "          end\n"
           << "        end\n"
           << "        if (loom_verbose_level >= 2) $display("
              "\"[loom][rtl][memory] port="
           << ordinal << " kind=%0d lanes=%0d element_bits=%0d\", "
           << port.prefix << "_request_kind, " << port.prefix
           << "_request_lane_count, " << port.prefix
           << "_request_element_width);\n"
           << "        loom_memory_response_data_" << ordinal
           << " <= loom_memory_response_next_" << ordinal << ";\n"
           << "        loom_memory_response_pending_" << ordinal << " <= 1;\n"
           << "      end\n";
  }
  output << "    end\n"
         << "  end\n";
}

void renderClocks(llvm::raw_ostream &output, llvm::ArrayRef<ClockPort> clocks) {
  for (const ClockPort &clock : clocks) {
    const std::uint64_t low = clock.periodFs / 2;
    const std::uint64_t high = clock.periodFs - low;
    output << "  initial begin\n"
           << "    " << clock.name << " = 0;\n"
           << "    #(" << clock.phaseFs + low << ");\n"
           << "    forever begin\n"
           << "      " << clock.name << " = 1;\n"
           << "      #(" << high << ");\n"
           << "      " << clock.name << " = 0;\n"
           << "      #(" << low << ");\n"
           << "    end\n"
           << "  end\n";
  }
}

void renderInputCounters(llvm::raw_ostream &output,
                         llvm::ArrayRef<const InputTokenStream *> inputs,
                         llvm::StringRef clock) {
  output << "  always_ff @(posedge " << clock << ") begin\n"
         << "    if (!loom_resets_released) begin\n";
  for (std::size_t ordinal = 0; ordinal != inputs.size(); ++ordinal)
    output << "      loom_input_index_" << ordinal << " <= 0;\n";
  output << "    end else begin\n";
  for (std::size_t ordinal = 0; ordinal != inputs.size(); ++ordinal)
    output << "      if (" << inputs[ordinal]->port.prefix << "_valid && "
           << inputs[ordinal]->port.prefix << "_ready) loom_input_index_"
           << ordinal << " <= loom_input_index_" << ordinal << " + 1;\n";
  output << "    end\n"
         << "  end\n";
}

/// Renders the AXI4-Lite configuration tasks of one configuration port. Both
/// tasks sample the handshake at the rising edge and drive the request
/// channels from it with nonblocking assignments, so a request becomes
/// visible after the edge and no delay control is needed to keep it off the
/// sampling edge. The write task presents the address and data channels
/// together, which AXI4-Lite permits and the configuration controller accepts
/// independently, and retires them per channel; the write still completes only
/// through its own B response before the caller continues, and the readback
/// and the atomic commit write are unchanged. A simulator evaluates the
/// configuration fan-out at every edge the driver acts on, so a driver that
/// acts only at the sampling edge spares the falling-edge evaluation of every
/// configuration cycle. Every read samples its accepted response at the next
/// falling edge. The caller can then present the next address before the next
/// rising edge, when the controller can replace the retiring response without
/// leaving the read channel idle.
void renderConfigurationTask(llvm::raw_ostream &output, llvm::StringRef prefix,
                             std::size_t ordinal, llvm::StringRef clock) {
  output << "  task automatic loom_cfg_write_" << ordinal << "(input logic ["
         << hardware::rtl::portableConfigurationAddressWidth - 1
         << ":0] address, input logic ["
         << hardware::rtl::portableConfigurationDataWidth - 1
         << ":0] data, input logic ["
         << hardware::rtl::portableConfigurationByteCount - 1
         << ":0] strobe, input logic falling_edge_response);\n"
         << "    integer wait_cycles;\n"
         << "    logic address_accepted;\n"
         << "    logic data_accepted;\n"
         << "    begin\n"
         << "      if (loom_verbose_level >= 3) $display(\"[loom][rtl][cfg] "
            "write address=%h data=%h strobe=%h\", address, data, strobe);\n"
         << "      " << prefix << "_awaddr <= address;\n"
         << "      " << prefix << "_awvalid <= 1;\n"
         << "      " << prefix << "_wdata <= data;\n"
         << "      " << prefix << "_wstrb <= strobe;\n"
         << "      " << prefix << "_wvalid <= 1;\n"
         << "      address_accepted = 0;\n"
         << "      data_accepted = 0;\n"
         << "      wait_cycles = 0;\n"
         << "      do begin\n"
         << "        @(posedge " << clock << ");\n"
         << "        address_accepted = address_accepted | " << prefix
         << "_awready;\n"
         << "        data_accepted = data_accepted | " << prefix << "_wready;\n"
         << "        if (address_accepted) " << prefix << "_awvalid <= 0;\n"
         << "        if (data_accepted) " << prefix << "_wvalid <= 0;\n"
         << "        wait_cycles = wait_cycles + 1;\n"
         << "        if (wait_cycles == " << kConfigurationHandshakeCycleLimit
         << " && (!address_accepted || !data_accepted)) $fatal(1, "
            "\"AXI4-Lite write handshake timed out\");\n"
         << "      end while (!address_accepted || !data_accepted);\n"
         << "      wait_cycles = 0;\n"
         << "      while (!" << prefix << "_bvalid) begin\n"
         << "        if (falling_edge_response) @(negedge " << clock
         << "); else @(posedge " << clock << ");\n"
         << "        wait_cycles = wait_cycles + 1;\n"
         << "        if (wait_cycles == " << kConfigurationHandshakeCycleLimit
         << " && !" << prefix
         << "_bvalid) $fatal(1, \"AXI4-Lite B response timed out\");\n"
         << "      end\n"
         << "      if (" << prefix
         << "_bresp !== 2'b00) $fatal(1, \"AXI4-Lite write failed\");\n"
         << "    end\n"
         << "  endtask\n"
         << "  task automatic loom_cfg_read_" << ordinal << "(input logic ["
         << hardware::rtl::portableConfigurationAddressWidth - 1
         << ":0] address, output logic ["
         << hardware::rtl::portableConfigurationDataWidth - 1
         << ":0] data, output logic [" << kAxiResponseWidth - 1
         << ":0] response);\n"
         << "    integer wait_cycles;\n"
         << "    begin\n"
         << "      " << prefix << "_araddr <= address;\n"
         << "      " << prefix << "_arvalid <= 1;\n"
         << "      wait_cycles = 0;\n"
         << "      do begin\n"
         << "        @(posedge " << clock << ");\n"
         << "        wait_cycles = wait_cycles + 1;\n"
         << "        if (wait_cycles == " << kConfigurationHandshakeCycleLimit
         << " && !" << prefix
         << "_arready) $fatal(1, \"AXI4-Lite AR handshake timed out\");\n"
         << "      end while (!" << prefix << "_arready);\n"
         << "      " << prefix << "_arvalid <= 0;\n"
         << "      wait_cycles = 0;\n"
         << "      do begin\n"
         << "        @(negedge " << clock << ");\n"
         << "        wait_cycles = wait_cycles + 1;\n"
         << "        if (wait_cycles == " << kConfigurationHandshakeCycleLimit
         << " && !" << prefix
         << "_rvalid) $fatal(1, \"AXI4-Lite R response timed out\");\n"
         << "      end while (!" << prefix << "_rvalid);\n"
         << "      data = " << prefix << "_rdata;\n"
         << "      response = " << prefix << "_rresp;\n"
         << "    end\n"
         << "  endtask\n";
}

void renderConfigurationProgram(llvm::raw_ostream &output,
                                const ConfigurationProgram &program,
                                std::size_t taskOrdinal) {
  const std::string wordIndex =
      "loom_cfg_program_word_" + std::to_string(taskOrdinal);
  const std::string words = "loom_cfg_program_" + std::to_string(taskOrdinal);
  // The configuration stages have very different simulation cost, so each
  // boundary is announced once at the ordinary verbosity level with its
  // simulation time and flushed. A host-side timestamp of the announcement is
  // then the stage boundary even under a simulator that buffers its standard
  // output. The time is printed as an integer of the harness's femtosecond
  // unit, because the simulators format `%t` in different units.
  output << "    if (loom_verbose_level >= 1) begin\n"
         << "      $display(\"[loom][rtl][stage] write_begin program="
         << taskOrdinal << " time_fs=%0d\", $time);\n"
         << "      $fflush();\n"
         << "    end\n"
         << "    for (" << wordIndex << " = 0; " << wordIndex << " < "
         << program.layout.payloadWordCount << "; " << wordIndex << " = "
         << wordIndex << " + 1) begin\n"
         << "      loom_cfg_write_" << taskOrdinal << "("
         << hardware::rtl::portableConfigurationAddressWidth << "'h"
         << llvm::format_hex_no_prefix(
                program.layout.baseAddress,
                hardware::rtl::portableConfigurationAddressWidth /
                    kBitsPerHexDigit)
         << " + " << wordIndex << " * "
         << hardware::rtl::portableConfigurationByteCount << ", " << words
         << "[" << wordIndex << "][31:0], " << words << "[" << wordIndex
         << "][35:32], 0);\n"
         << "      loom_cfg_payload_writes_" << taskOrdinal << " = "
         << "loom_cfg_payload_writes_" << taskOrdinal << " + 1;\n"
         << "    end\n";
  output << "    loom_cfg_write_" << taskOrdinal << "("
         << hardware::rtl::portableConfigurationAddressWidth << "'h"
         << llvm::format_hex_no_prefix(
                program.layout.commitAddress,
                hardware::rtl::portableConfigurationAddressWidth /
                    kBitsPerHexDigit)
         << ", " << hardware::rtl::portableConfigurationDataWidth
         << "'h00000001, " << hardware::rtl::portableConfigurationByteCount
         << "'h1, 1);\n"
         << "    loom_cfg_atomic_commits_" << taskOrdinal << " = "
         << "loom_cfg_atomic_commits_" << taskOrdinal << " + 1;\n";
  // The three configuration stages have very different simulation cost, so
  // each boundary is announced once at the ordinary verbosity level with its
  // simulation time and flushed, so a host-side timestamp of the announcement
  // is the stage boundary even under a simulator that buffers its standard
  // output, and the simulation time gives the stage its exact cycle count.
  // The time is printed as an integer of the harness's femtosecond unit,
  // because the simulators format `%t` in different units.
  output << "    if (loom_verbose_level >= 1) begin\n"
         << "      $display(\"[loom][rtl][stage] readback_begin program="
         << taskOrdinal << " time_fs=%0d\", $time);\n"
         << "      $fflush();\n"
         << "    end\n";
  output << "    for (" << wordIndex << " = 0; " << wordIndex << " < "
         << program.layout.payloadWordCount << "; " << wordIndex << " = "
         << wordIndex << " + 1) begin\n"
         << "      loom_cfg_read_" << taskOrdinal << "("
         << hardware::rtl::portableConfigurationAddressWidth << "'h"
         << llvm::format_hex_no_prefix(
                program.layout.baseAddress,
                hardware::rtl::portableConfigurationAddressWidth /
                    kBitsPerHexDigit)
         << " + " << wordIndex << " * "
         << hardware::rtl::portableConfigurationByteCount
         << ", loom_cfg_readback, loom_cfg_response);\n"
         << "      if (loom_cfg_response !== 2'b00 || "
            "loom_cfg_readback !== "
         << words << "[" << wordIndex
         << "][31:0]) $fatal(1, \"active configuration readback mismatch\");\n"
         << "      loom_cfg_active_word_comparisons_" << taskOrdinal << " = "
         << "loom_cfg_active_word_comparisons_" << taskOrdinal << " + 1;\n"
         << "    end\n";
  output << "    loom_cfg_read_" << taskOrdinal << "("
         << hardware::rtl::portableConfigurationAddressWidth << "'h"
         << llvm::format_hex_no_prefix(
                program.layout.statusAddress,
                hardware::rtl::portableConfigurationAddressWidth /
                    kBitsPerHexDigit)
         << ", loom_cfg_readback, loom_cfg_response);\n"
         << "    if (loom_cfg_response !== 2'b00 || "
            "loom_cfg_readback !== "
         << hardware::rtl::portableConfigurationDataWidth
         << "'h00000000) $fatal(1, "
            "\"configuration status did not clear\");\n"
         << "    loom_cfg_passing_status_reads_" << taskOrdinal << " = "
         << "loom_cfg_passing_status_reads_" << taskOrdinal << " + 1;\n";
}

void renderConfigurationTransportReceiptWriter(
    llvm::raw_ostream &output, const MappedRtlInvocationFacts &facts,
    llvm::StringRef receiptPath) {
  output << "  task automatic loom_write_configuration_transport_receipt;\n"
         << "    integer receipt_file;\n"
         << "    begin\n"
         << "      if (loom_cfg_receipt_written) $fatal(1, \"configuration "
            "transport receipt was written more than once\");\n";
  for (const auto &[ordinal, program] :
       llvm::enumerate(facts.configurationPrograms)) {
    output << "      if (loom_cfg_payload_writes_" << ordinal << " != "
           << program.layout.payloadWordCount
           << ") $fatal(1, \"configuration payload write count is "
              "incomplete\");\n"
           << "      if (loom_cfg_atomic_commits_" << ordinal
           << " != 1) $fatal(1, \"configuration atomic commit count is "
              "not one\");\n"
           << "      if (loom_cfg_active_word_comparisons_" << ordinal
           << " != " << program.layout.payloadWordCount
           << ") $fatal(1, \"configuration active-word comparison count is "
              "incomplete\");\n"
           << "      if (loom_cfg_passing_status_reads_" << ordinal
           << " != 1) $fatal(1, \"configuration passing status read count "
              "is not one\");\n";
  }
  output << "      receipt_file = $fopen(\"" << receiptPath << "\", \"w\");\n"
         << "      if (receipt_file == 0) $fatal(1, \"could not open "
            "configuration transport receipt\");\n"
         << "      $fwrite(receipt_file, \""
         << mappedRtlConfigurationTransportReceiptSchema << " "
         << mappedRtlConfigurationTransportReceiptVersion << "\\n\");\n"
         << "      $fwrite(receipt_file, \"programs "
         << facts.configurationPrograms.size() << "\\n\");\n";
  for (std::size_t ordinal = 0;
       ordinal != facts.configurationPrograms.size(); ++ordinal)
    output << "      $fwrite(receipt_file, \"program " << ordinal
           << " payload_writes %0d atomic_commits %0d "
              "active_word_comparisons %0d passing_status_reads %0d\\n\", "
           << "loom_cfg_payload_writes_" << ordinal << ", "
           << "loom_cfg_atomic_commits_" << ordinal << ", "
           << "loom_cfg_active_word_comparisons_" << ordinal << ", "
           << "loom_cfg_passing_status_reads_" << ordinal << ");\n";
  output << "      $fwrite(receipt_file, \"end\\n\");\n"
         << "      $fflush(receipt_file);\n"
         << "      $fclose(receipt_file);\n"
         << "      loom_cfg_receipt_written = 1;\n"
         << "    end\n"
         << "  endtask\n";
}

void renderResultWriter(llvm::raw_ostream &output,
                        const MappedRtlInvocationFacts &facts,
                        llvm::StringRef resultPath) {
  output << "  task automatic loom_write_result(input bit stopped);\n"
         << "    integer result_file;\n"
         << "    integer token_ordinal;\n"
         << "    begin\n"
         << "      result_file = $fopen(\"" << resultPath << "\", \"w\");\n"
         << "      if (result_file == 0) $fatal(1, \"could not open mapped RTL "
            "result\");\n"
         << "      $fwrite(result_file, \"" << mappedRtlResultSchema << " "
         << mappedRtlResultVersion << "\\n\");\n"
         << "      if (stopped) $fwrite(result_file, \"terminal "
         << mappedRtlTerminalStatusSpelling(
                MappedRtlTerminalStatus::StoppedByLimit)
         << "\\n\");\n"
         << "      else $fwrite(result_file, \"terminal "
         << mappedRtlTerminalStatusSpelling(MappedRtlTerminalStatus::Retired)
         << "\\n\");\n"
         << "      $fwrite(result_file, \"launch_cycle %0d\\n\", "
            "loom_launch_cycle);\n"
         << "      if (stopped) $fwrite(result_file, "
            "\"retirement_cycle absent\\n\");\n"
         << "      else $fwrite(result_file, \"retirement_cycle %0d\\n\", "
            "loom_retirement_cycle);\n"
         << "      $fwrite(result_file, \"terminal_cycle %0d\\n\", "
            "loom_cycle);\n"
         << "      $fwrite(result_file, \"value_results "
         << facts.valueResults.size() << "\\n\");\n";
  for (const auto &[ordinal, value] : llvm::enumerate(facts.valueResults)) {
    output << "      if (loom_value_result_" << ordinal
           << ".size() == 0) $fwrite(result_file, \"value " << ordinal
           << " absent\\n\");\n"
           << "      else begin\n"
           << "        if (loom_value_result_" << ordinal
           << ".size() != 1) $fatal(1, \"value result published more than "
              "once\");\n"
           << "        $fwrite(result_file, \"value " << ordinal << " "
           << value.tokenBitWidth << " b%b\\n\", loom_value_result_" << ordinal
           << "[0]);\n"
           << "      end\n";
  }
  output << "      $fwrite(result_file, \"stream_outputs "
         << facts.streamOutputs.size() << "\\n\");\n";
  for (const auto &[ordinal, stream] : llvm::enumerate(facts.streamOutputs)) {
    output << "      $fwrite(result_file, \"stream " << ordinal << " "
           << mappedRtlStreamTerminationSpelling(
                  sim::StreamTermination::ClosedAfterLast)
           << " " << stream.tokenBitWidth << " %0d\", loom_stream_output_"
           << ordinal << ".size());\n"
           << "      for (token_ordinal = 0; token_ordinal < "
              "loom_stream_output_"
           << ordinal << ".size(); token_ordinal = token_ordinal + 1)\n"
           << "        $fwrite(result_file, \" b%b\", loom_stream_output_"
           << ordinal << "[token_ordinal]);\n"
           << "      $fwrite(result_file, \"\\n\");\n";
  }
  output << "      $fwrite(result_file, \"memories "
         << facts.memoryObservations.size() << "\\n\");\n";
  for (const auto &[ordinal, observation] :
       llvm::enumerate(facts.memoryObservations)) {
    const RuntimeMemoryImage &image =
        facts.memoryImages[observation.objectOrdinal];
    const std::uint64_t count =
        image.initialBytes.size() - observation.byteOffset;
    const std::uint64_t begin =
        image.canonicalBaseAddress + observation.byteOffset;
    output << "      $fwrite(result_file, \"memory " << ordinal << " " << count
           << "\");\n"
           << "      for (token_ordinal = 0; token_ordinal < " << count
           << "; token_ordinal = token_ordinal + 1)\n"
           << "        $fwrite(result_file, \" d%02x\", "
              "loom_runtime_memory["
           << begin << " + token_ordinal]);\n"
           << "      $fwrite(result_file, \"\\n\");\n";
  }
  output << "      $fwrite(result_file, \"end\\n\");\n"
         << "      $fclose(result_file);\n"
         << "    end\n"
         << "  endtask\n";
}

} // namespace

llvm::Expected<std::string>
renderMappedRtlConfigurationProgramFile(const ConfigurationProgram &program) {
  if (llvm::Error error = validateProgram(program))
    return std::move(error);
  std::string text;
  llvm::raw_string_ostream output(text);
  for (std::uint64_t word = 0; word != program.layout.payloadWordCount; ++word)
    output << llvm::format_hex_no_prefix(imageStrobe(program.image, word), 1)
           << llvm::format_hex_no_prefix(imageWord(program.image, word), 8)
           << '\n';
  return text;
}

llvm::Expected<std::string>
renderMappedRtlTestbench(const MappedRtlInvocationFacts &facts,
                         llvm::ArrayRef<std::string> configurationProgramPaths,
                         llvm::StringRef resultPath,
                         llvm::StringRef configurationTransportReceiptPath) {
  if (facts.top.empty())
    return invalid("RTL top name is absent");
  if (facts.selectedClock.empty())
    return invalid("selected clock is absent");
  if (facts.selectedClockPeriodFs == 0)
    return invalid("selected clock period is zero");
  if (facts.cycleLimit == 0)
    return invalid("cycle limit is zero");
  if (!facts.startInput)
    return invalid("graph start transport is absent");
  if (facts.completionOutputs.empty())
    return invalid("graph completion transports are absent");
  for (const ClockPort &clock : facts.clockPorts)
    if (clock.periodFs < kMinimumClockPeriodFs ||
        clock.phaseFs >= clock.periodFs)
      return invalid("Clock contract cannot be represented at 1 fs precision");
  for (const ConfigurationProgram &program : facts.configurationPrograms)
    if (llvm::Error error = validateProgram(program))
      return std::move(error);
  if (configurationProgramPaths.size() != facts.configurationPrograms.size())
    return invalid("configuration program path count is inconsistent");
  for (llvm::StringRef path : configurationProgramPaths) {
    const std::filesystem::path parsed(path.str());
    if (path.empty() || path.contains('\\') || path.contains('\0') ||
        path.contains('"') || parsed.is_absolute() ||
        parsed.lexically_normal() != parsed)
      return invalid("configuration program path is not canonical");
  }
  auto projectedMemoryExtent = memoryExtent(facts);
  if (!projectedMemoryExtent)
    return projectedMemoryExtent.takeError();
  for (const MemoryBoundaryPort &port : facts.memoryBoundaryPorts) {
    if (port.addressBitWidth == 0 || port.dataBitWidth == 0 ||
        port.dataBitWidth % kBitsPerByte != 0 || port.maskBitWidth == 0 ||
        port.bindings.empty())
      return invalid("memory boundary plan is incomplete");
    for (std::size_t ordinal = 0; ordinal != port.bindings.size(); ++ordinal) {
      const MemoryBoundaryBinding &binding = port.bindings[ordinal];
      if (binding.rootObjectOrdinal >= facts.memoryImages.size() ||
          binding.rootByteOffset >=
              facts.memoryImages[binding.rootObjectOrdinal]
                  .initialBytes.size() ||
          (ordinal != 0 &&
           port.bindings[ordinal - 1].requestContext >= binding.requestContext))
        return invalid("memory boundary binding plan is not canonical");
    }
  }
  for (const MemoryObservationPlan &observation : facts.memoryObservations)
    if (observation.objectOrdinal >= facts.memoryImages.size() ||
        observation.byteOffset >=
            facts.memoryImages[observation.objectOrdinal].initialBytes.size())
      return invalid("memory observation plan is incomplete");

  std::vector<const InputTokenStream *> inputs;
  inputs.push_back(&*facts.startInput);
  for (const InputTokenStream &input : facts.valueInputs)
    inputs.push_back(&input);
  for (const InputTokenStream &input : facts.streamInputs)
    inputs.push_back(&input);

  std::string text;
  llvm::raw_string_ostream output(text);
  output << "`timescale 1fs/1fs\n"
         << "module " << mappedRtlHarnessTop << "(\n"
         << "  output wire loom_engine_retired,\n"
         << "  output wire [63:0] loom_engine_launch_cycle,\n"
         << "  output wire [63:0] loom_engine_retirement_cycle\n"
         << ");\n";
  for (const RtlPort &port : facts.rootPorts)
    renderPortDeclaration(output, port);
  output << "  logic loom_inputs_enabled;\n"
         << "  logic loom_resets_released;\n"
         << "  logic loom_retired;\n"
         << "  logic loom_cfg_receipt_written;\n"
         << "  longint unsigned loom_cycle;\n"
         << "  longint unsigned loom_launch_cycle;\n"
         << "  longint unsigned loom_retirement_cycle;\n"
         << "  logic [" << hardware::rtl::portableConfigurationDataWidth - 1
         << ":0] loom_cfg_readback;\n"
         << "  logic [" << kAxiResponseWidth - 1 << ":0] loom_cfg_response;\n"
         << "  integer loom_verbose_level;\n";
  for (const auto &[ordinal, program] :
       llvm::enumerate(facts.configurationPrograms))
    output << "  logic [" << kConfigurationProgramWordWidth - 1
           << ":0] loom_cfg_program_" << ordinal << " [0:"
           << program.layout.payloadWordCount - 1 << "];\n"
           << "  integer loom_cfg_program_word_" << ordinal << ";\n"
           << "  longint unsigned loom_cfg_payload_writes_" << ordinal
           << ";\n"
           << "  longint unsigned loom_cfg_atomic_commits_" << ordinal
           << ";\n"
           << "  longint unsigned loom_cfg_active_word_comparisons_" << ordinal
           << ";\n"
           << "  longint unsigned loom_cfg_passing_status_reads_" << ordinal
           << ";\n";
  output << "  assign loom_engine_retired = loom_retired;\n"
         << "  assign loom_engine_launch_cycle = loom_launch_cycle;\n"
         << "  assign loom_engine_retirement_cycle = "
            "loom_retirement_cycle;\n";
  for (const auto &[ordinal, value] : llvm::enumerate(facts.valueResults))
    output << "  logic [" << value.tokenBitWidth - 1 << ":0] loom_value_result_"
           << ordinal << " [$];\n";
  for (const auto &[ordinal, stream] : llvm::enumerate(facts.streamOutputs))
    output << "  logic [" << stream.tokenBitWidth - 1
           << ":0] loom_stream_output_" << ordinal << " [$];\n";
  for (const InputTokenStream &input : facts.streamInputs) {
    const std::uint64_t ordinal = *input.runtimeStreamOrdinal;
    output << "  logic loom_runtime_stream_enabled_" << ordinal << ";\n"
           << "  logic [" << input.port.payloadBitWidth - 1
           << ":0] loom_runtime_stream_token_" << ordinal << ";\n"
           << "  logic [" << input.port.payloadBitWidth - 1
           << ":0] loom_runtime_stream_" << ordinal << " [$];\n"
           << "  string loom_runtime_stream_path_" << ordinal << ";\n"
           << "  integer loom_runtime_stream_file_" << ordinal << ";\n"
           << "  integer loom_runtime_stream_scan_" << ordinal << ";\n";
  }
  renderMemoryDeclarations(output, facts, *projectedMemoryExtent);

  output << "  " << facts.top << " dut (\n";
  for (std::size_t ordinal = 0; ordinal != facts.rootPorts.size(); ++ordinal) {
    const RtlPort &port = facts.rootPorts[ordinal];
    output << "    ." << port.name << "(" << port.name << ")"
           << (ordinal + 1 == facts.rootPorts.size() ? "\n" : ",\n");
  }
  output << "  );\n";
  renderClocks(output, facts.clockPorts);

  const std::set<std::string> custom = customDrivenPorts(facts);
  output << "  initial begin\n"
         << "    loom_inputs_enabled = 0;\n"
         << "    loom_resets_released = 0;\n"
         << "    loom_cfg_receipt_written = 0;\n"
         << "    if (!$value$plusargs(\"LOOM_VERBOSE_LEVEL=%d\", "
            "loom_verbose_level)) loom_verbose_level = 0;\n";
  for (std::size_t ordinal = 0;
       ordinal != facts.configurationPrograms.size(); ++ordinal)
    output << "    loom_cfg_payload_writes_" << ordinal << " = 0;\n"
           << "    loom_cfg_atomic_commits_" << ordinal << " = 0;\n"
           << "    loom_cfg_active_word_comparisons_" << ordinal << " = 0;\n"
           << "    loom_cfg_passing_status_reads_" << ordinal << " = 0;\n";
  for (const auto &[ordinal, path] :
       llvm::enumerate(configurationProgramPaths))
    output << "    $readmemh(\"" << path << "\", loom_cfg_program_"
           << ordinal << ");\n";
  for (const InputTokenStream &input : facts.streamInputs) {
    const std::uint64_t ordinal = *input.runtimeStreamOrdinal;
    output << "    loom_runtime_stream_enabled_" << ordinal << " = 0;\n"
           << "    if ($value$plusargs(\"LOOM_STREAM_INPUT_" << ordinal
           << "=%s\", loom_runtime_stream_path_" << ordinal << ")) begin\n"
           << "      loom_runtime_stream_file_" << ordinal
           << " = $fopen(loom_runtime_stream_path_" << ordinal << ", \"r\");\n"
           << "      if (loom_runtime_stream_file_" << ordinal
           << " == 0) $fatal(1, \"could not open runtime stream input\");\n"
           << "      while (!$feof(loom_runtime_stream_file_" << ordinal
           << ")) begin\n"
           << "        loom_runtime_stream_scan_" << ordinal
           << " = $fscanf(loom_runtime_stream_file_" << ordinal
           << ", \"%b\\n\", loom_runtime_stream_token_" << ordinal << ");\n"
           << "        if (loom_runtime_stream_scan_" << ordinal
           << " == 1) loom_runtime_stream_" << ordinal
           << ".push_back(loom_runtime_stream_token_" << ordinal << ");\n"
           << "        else if (!$feof(loom_runtime_stream_file_" << ordinal
           << ")) $fatal(1, \"runtime stream input is malformed\");\n"
           << "      end\n"
           << "      $fclose(loom_runtime_stream_file_" << ordinal << ");\n"
           << "      loom_runtime_stream_enabled_" << ordinal << " = 1;\n"
           << "    end\n";
  }
  renderMemoryInitialization(output, facts, *projectedMemoryExtent);
  for (const RtlPort &port : facts.rootPorts) {
    if (port.direction == hardware::RepresentationSignalDirection::Output ||
        custom.count(port.name))
      continue;
    output << "    " << port.name << " = "
           << (llvm::StringRef(port.name).ends_with("_ready") ? "1" : "'0")
           << ";\n";
  }
  for (const ConfigurationProgram &program : facts.configurationPrograms) {
    output << "    " << program.portPrefix << "_awaddr = 0;\n"
           << "    " << program.portPrefix << "_awvalid = 0;\n"
           << "    " << program.portPrefix << "_wdata = 0;\n"
           << "    " << program.portPrefix << "_wstrb = 0;\n"
           << "    " << program.portPrefix << "_wvalid = 0;\n"
           << "    " << program.portPrefix << "_bready = 1;\n"
           << "    " << program.portPrefix << "_araddr = 0;\n"
           << "    " << program.portPrefix << "_arvalid = 0;\n"
           << "    " << program.portPrefix << "_rready = 1;\n";
  }
  for (const ResetPort &reset : facts.resetPorts)
    output << "    " << reset.name << " = " << reset.assertedValue << ";\n";
  // The reset release, the harness enables, and every configuration request
  // are driven at the rising edge with nonblocking assignments, so the design
  // observes each of them from the following edge. The first request is
  // presented in the same edge as the reset release, so the design observes
  // both from the next edge; together with the falling-edge response waits of
  // the commit write and the status read this keeps every design-observed
  // configuration edge and the kernel launch edge where a falling-edge driver
  // put them.
  output << "    repeat (" << kResetReleaseCycleCount << ") @(posedge "
         << facts.selectedClock << ");\n";
  for (const ResetPort &reset : facts.resetPorts)
    output << "    " << reset.name << " <= " << !reset.assertedValue << ";\n";
  output << "    loom_resets_released <= 1;\n";
  for (const auto &[ordinal, program] :
       llvm::enumerate(facts.configurationPrograms))
    renderConfigurationProgram(output, program, ordinal);
  output << "    loom_write_configuration_transport_receipt();\n"
         << "    if (loom_verbose_level >= 1) begin\n"
         << "      $display(\"[loom][rtl][stage] kernel_begin time_fs=%0d\", "
            "$time);\n"
         << "      $fflush();\n"
         << "    end\n"
         << "    loom_inputs_enabled <= 1;\n"
         << "    while (!loom_retired && loom_cycle < " << facts.cycleLimit
         << ") @(posedge " << facts.selectedClock << ");\n"
         << "    if (loom_retired) begin\n"
         << "      @(negedge " << facts.selectedClock << ");\n"
         << "      loom_write_result(0);\n"
         << "    end else begin\n"
         << "      loom_write_result(1);\n"
         << "    end\n"
         << "    $finish;\n"
         << "  end\n";

  for (const auto &[ordinal, input] : llvm::enumerate(inputs)) {
    const std::string index = "loom_input_index_" + std::to_string(ordinal);
    output << "  longint unsigned " << index << ";\n";
    output << "  always_comb begin\n"
           << "    " << input->port.prefix
           << "_valid = loom_inputs_enabled && ";
    if (input->runtimeStreamOrdinal)
      output << index << " < (loom_runtime_stream_enabled_"
             << *input->runtimeStreamOrdinal << " ? loom_runtime_stream_"
             << *input->runtimeStreamOrdinal
             << ".size() : " << input->tokenCount << ");\n";
    else
      output << index << " < " << input->tokenCount << ";\n";
    if (input->port.payloadBitWidth != 0) {
      output << "    " << input->port.prefix << "_data = '0;\n";
      if (input->runtimeStreamOrdinal)
        output << "    if (loom_runtime_stream_enabled_"
               << *input->runtimeStreamOrdinal << ") " << input->port.prefix
               << "_data = loom_runtime_stream_" << *input->runtimeStreamOrdinal
               << "[" << index << "];\n"
               << "    else begin\n";
      if (!input->tokens.empty()) {
        output << "    case (" << index << ")\n";
        for (const auto &[tokenOrdinal, token] :
             llvm::enumerate(input->tokens)) {
          llvm::APInt physical = token.zext(input->port.payloadBitWidth);
          output << "      " << tokenOrdinal << ": " << input->port.prefix
                 << "_data = " << input->port.payloadBitWidth << "'h"
                 << hexBits(physical) << ";\n";
        }
        output << "      default: ;\n"
               << "    endcase\n";
      }
      if (input->runtimeStreamOrdinal)
        output << "    end\n";
    }
    if (input->port.physicalTag)
      output << "    " << input->port.prefix
             << "_tag = " << input->port.physicalTag->getBitWidth() << "'h"
             << hexBits(*input->port.physicalTag) << ";\n";
    output << "  end\n";
  }
  renderInputCounters(output, inputs, facts.selectedClock);
  for (const auto &[ordinal, input] : llvm::enumerate(inputs)) {
    output << "  always_ff @(posedge " << facts.selectedClock << ") begin\n"
           << "    if (loom_inputs_enabled && loom_verbose_level >= 3) "
              "$display(\"[loom][rtl][input] ordinal="
           << ordinal << " cycle=%0d valid=%0d ready=%0d index=%0d\", "
           << "loom_cycle, " << input->port.prefix << "_valid, "
           << input->port.prefix << "_ready, loom_input_index_" << ordinal
           << ");\n"
           << "    if (loom_inputs_enabled && loom_verbose_level >= 2 && "
           << input->port.prefix << "_valid && " << input->port.prefix
           << "_ready) $display(\"[loom][rtl][input] ordinal=" << ordinal
           << " accepted_cycle=%0d\", loom_cycle);\n"
           << "  end\n";
  }

  for (const auto &[ordinal, program] :
       llvm::enumerate(facts.configurationPrograms))
    renderConfigurationTask(output, program.portPrefix, ordinal,
                            facts.selectedClock);
  renderConfigurationTransportReceiptWriter(
      output, facts, configurationTransportReceiptPath);
  renderMemoryService(output, facts);

  for (const auto &[ordinal, value] : llvm::enumerate(facts.valueResults))
    renderPhysicalTagAssertion(output, value.port, facts.selectedClock,
                               "value result", ordinal);
  for (const auto &[ordinal, stream] : llvm::enumerate(facts.streamOutputs))
    renderPhysicalTagAssertion(output, stream.port, facts.selectedClock,
                               "stream output", ordinal);
  for (const auto &[ordinal, completion] :
       llvm::enumerate(facts.completionOutputs))
    renderPhysicalTagAssertion(output, completion, facts.selectedClock,
                               "completion", ordinal);

  for (const auto &[ordinal, value] : llvm::enumerate(facts.valueResults))
    output << "  always_ff @(posedge " << facts.selectedClock << ")\n"
           << "    if (loom_resets_released && " << value.port.prefix
           << "_valid && " << value.port.prefix << "_ready) loom_value_result_"
           << ordinal << ".push_back(" << value.port.prefix << "_data["
           << value.tokenBitWidth - 1 << ":0]);\n";
  for (const auto &[ordinal, stream] : llvm::enumerate(facts.streamOutputs))
    output << "  always_ff @(posedge " << facts.selectedClock << ")\n"
           << "    if (loom_resets_released && " << stream.port.prefix
           << "_valid && " << stream.port.prefix
           << "_ready) loom_stream_output_" << ordinal << ".push_back("
           << stream.port.prefix << "_data[" << stream.tokenBitWidth - 1
           << ":0]);\n";

  output << "  logic [" << facts.completionOutputs.size() - 1
         << ":0] loom_completion_seen;\n"
         << "  logic [" << facts.completionOutputs.size() - 1
         << ":0] loom_completion_fire;\n"
         << "  always_comb begin\n";
  for (const auto &[ordinal, completion] :
       llvm::enumerate(facts.completionOutputs))
    output << "    loom_completion_fire[" << ordinal
           << "] = " << completion.prefix << "_valid && " << completion.prefix
           << "_ready;\n";
  output << "  end\n"
         << "  always_ff @(posedge " << facts.selectedClock << ") begin\n"
         << "    if (!loom_resets_released) begin\n"
         << "      loom_cycle <= 0;\n"
         << "      loom_launch_cycle <= 0;\n"
         << "      loom_retirement_cycle <= 0;\n"
         << "      loom_completion_seen <= '0;\n"
         << "      loom_retired <= 0;\n"
         << "    end else if (loom_inputs_enabled) begin\n"
         << "      if (loom_verbose_level >= 3) $display("
            "\"[loom][rtl][progress] cycle=%0d completion_seen=%b "
            "completion_fire=%b\", loom_cycle, loom_completion_seen, "
            "loom_completion_fire);\n"
         << "      loom_cycle <= loom_cycle + 1;\n"
         << "      loom_completion_seen <= loom_completion_seen | "
            "loom_completion_fire;\n"
         << "      if (" << facts.startInput->port.prefix << "_valid && "
         << facts.startInput->port.prefix
         << "_ready) loom_launch_cycle <= loom_cycle;\n"
         << "      if (!loom_retired && &(loom_completion_seen | "
            "loom_completion_fire)) begin\n"
         << "        if (loom_verbose_level >= 1) $display("
            "\"[loom][rtl][progress] retired_cycle=%0d\", loom_cycle);\n"
         << "        loom_retired <= 1;\n"
         << "        loom_retirement_cycle <= loom_cycle;\n"
         << "      end\n"
         << "    end\n"
         << "  end\n";
  renderResultWriter(output, facts, resultPath);
  output << "endmodule\n";
  return text;
}

llvm::Expected<std::string> renderMappedRtlVerilatorDriver(
    const MappedRtlInvocationFacts &facts, const MappedRtlVerilationPlan &plan,
    MappedRtlVerilationStyle style,
    llvm::ArrayRef<std::string> hierarchyMakeVariables,
    llvm::StringRef testbenchPath, llvm::StringRef simulatorExecutablePath,
    std::optional<llvm::StringRef> bridgeEngineSourcePath) {
  if (facts.rtlPaths.empty() && facts.rtlLibraryDirectories.empty())
    return invalid("Verilator driver has no RTL sources");
  if (plan.buildJobs == 0 || plan.modelThreads == 0)
    return invalid("Verilator parallelism must be positive");
  if (style == MappedRtlVerilationStyle::Flat &&
      !hierarchyMakeVariables.empty())
    return invalid("flat Verilation has no hierarchical make variables");
  std::string text;
  llvm::raw_string_ostream output(text);
  const std::filesystem::path simulatorExecutable(
      simulatorExecutablePath.str());
  // `-j` is the Verilation job count and the job count of the make that
  // Verilator runs for hierarchical Verilation. `--threads` and
  // `--hierarchical-threads` carry one model thread count so the generated
  // main, the root model, and the hierarchical schedule agree; the flat style
  // has no hierarchical schedule.
  output << "--cc\n--exe\n";
  if (!bridgeEngineSourcePath)
    output << "--main\n";
  if (style == MappedRtlVerilationStyle::Hierarchical)
    output << "--hierarchical\n";
  output << "-j\n"
         << plan.buildJobs << "\n--threads\n"
         << plan.modelThreads << "\n";
  if (style == MappedRtlVerilationStyle::Hierarchical)
    output << "--hierarchical-threads\n" << plan.modelThreads << "\n";
  output << "--timing\n--Wall\n--Wno-fatal\n"
            "--Wno-DECLFILENAME\n--Wno-UNUSEDSIGNAL\n--Wno-PINMISSING\n"
            "--Wno-TIMESCALEMOD\n"
            "--compiler\nclang\n-CFLAGS\n-std=c++20\n--top-module\n"
         << mappedRtlHarnessTop << "\n--Mdir\n"
         << simulatorExecutable.parent_path().generic_string() << "\n-o\n"
         << simulatorExecutable.filename().generic_string() << "\n";
  for (const std::string &variable : hierarchyMakeVariables)
    output << "-MAKEFLAGS\n" << variable << "\n";
  for (const std::string &path : facts.rtlPaths)
    output << path << "\n";
  for (const std::string &path : facts.rtlLibraryDirectories)
    output << "-y\n" << path << "\n+libext+.sv\n";
  output << testbenchPath << "\n";
  if (bridgeEngineSourcePath)
    output << *bridgeEngineSourcePath << "\n";
  return text;
}

llvm::Expected<std::string>
renderMappedRtlVcsDriver(const MappedRtlInvocationFacts &facts,
                         const MappedRtlVcsCompilationPlan &plan,
                         llvm::StringRef testbenchPath,
                         llvm::StringRef workDirectoryPath,
                         llvm::StringRef simulatorExecutablePath) {
  if (facts.rtlPaths.empty())
    return invalid("VCS driver has no RTL sources");
  if (plan.buildJobs == 0)
    return invalid("VCS parallel compilation count must be positive");
  std::string text;
  llvm::raw_string_ostream output(text);
  // The harness declares its own 1 fs timescale; the same scale is applied to
  // every RTL module so the clock periods of the harness are exact.
  output << "-sverilog\n-timescale=1fs/1fs\n-top\n"
         << mappedRtlHarnessTop << "\n-j" << plan.buildJobs
         << "\n-Mdir=" << workDirectoryPath << "/csrc\n-o\n"
         << simulatorExecutablePath << "\n";
  for (const std::string &path : facts.rtlPaths)
    output << path << "\n";
  output << testbenchPath << "\n";
  return text;
}

llvm::Expected<std::string>
renderMappedRtlXceliumDriver(const MappedRtlInvocationFacts &facts,
                             llvm::StringRef testbenchPath,
                             llvm::StringRef libraryDirectoryPath) {
  if (facts.rtlPaths.empty())
    return invalid("Xcelium driver has no RTL sources");
  std::string text;
  llvm::raw_string_ostream output(text);
  // SystemVerilog and the harness's own 1 fs timescale for every RTL module,
  // as for VCS; the snapshot library lies one level below the bundle's
  // `work/` root, and no log, key, or history file is written beside it.
  output << "-sv\n-timescale\n1fs/1fs\n-top\n"
         << mappedRtlHarnessTop << "\n-xmlibdirname\n"
         << libraryDirectoryPath << "\n-nolog\n-nokey\n-nohistory\n";
  for (const std::string &path : facts.rtlPaths)
    output << path << "\n";
  output << testbenchPath << "\n";
  return text;
}

llvm::Expected<sim::SpatialFunctionalObservations>
projectMappedRtlFunctionalObservations(
    const MappedRtlObservationFacts &facts,
    const MappedRtlSimulationResult &result) {
  const auto *workload = facts.inputs->workload.spatial();
  if (!workload)
    return invalid("invocation lost its Spatial workload");
  auto program = facts.inputs->dataflow.view();
  if (!program)
    return program.takeError();
  auto shapes = sim::projectSpatialSimulationBoundaryShapes(
      *program, workload->launchRef);
  if (!shapes)
    return shapes.takeError();
  if (result.valueResults.size() !=
          workload->observableContract.valueResults.size() ||
      result.streamOutputs.size() !=
          workload->observableContract.streamOutputs.size() ||
      result.memories.size() != facts.memoryObservations.size() ||
      result.memories.size() != workload->observableContract.memories.size())
    return invalid(
        "RTL result cardinality disagrees with the observable contract");

  sim::SpatialFunctionalObservations observations;
  observations.valueResults.reserve(result.valueResults.size());
  for (const auto &[index, resultValue] :
       llvm::enumerate(result.valueResults)) {
    const std::uint64_t ordinal =
        workload->observableContract.valueResults[index];
    const sim::SpatialSimulationValueShape shape =
        shapes->valueResults[ordinal];
    if (!resultValue.token) {
      observations.valueResults.emplace_back(sim::NotPublishedValueResult{});
      continue;
    }
    auto lanes =
        sim::unpackDefinedSpatialSimulationToken(*resultValue.token, shape);
    if (!lanes)
      return lanes.takeError();
    observations.valueResults.emplace_back(sim::PublishedValueResult{
        sim::CanonicalValueSequence{1, std::move(*lanes)}});
  }
  observations.streamOutputs.reserve(result.streamOutputs.size());
  for (const auto &[index, resultStream] :
       llvm::enumerate(result.streamOutputs)) {
    const std::uint64_t ordinal =
        workload->observableContract.streamOutputs[index];
    const sim::SpatialSimulationValueShape shape =
        shapes->streamOutputs[ordinal];
    std::vector<sim::SemanticLane> lanes;
    for (const llvm::APInt &token : resultStream.tokens) {
      auto unpacked = sim::unpackDefinedSpatialSimulationToken(token, shape);
      if (!unpacked)
        return unpacked.takeError();
      lanes.insert(lanes.end(), std::make_move_iterator(unpacked->begin()),
                   std::make_move_iterator(unpacked->end()));
    }
    observations.streamOutputs.push_back(sim::CanonicalStreamSequence{
        sim::CanonicalValueSequence{resultStream.tokens.size(),
                                    std::move(lanes)},
        resultStream.termination});
  }
  observations.memories.reserve(result.memories.size());
  for (std::size_t ordinal = 0; ordinal != result.memories.size(); ++ordinal) {
    const MappedRtlMemoryObservation &resultMemory = result.memories[ordinal];
    const MemoryObservationPlan &plan = facts.memoryObservations[ordinal];
    if (plan.objectOrdinal >= facts.memoryImages.size())
      return invalid("memory result names an absent runtime object");
    const RuntimeMemoryImage &image = facts.memoryImages[plan.objectOrdinal];
    if (plan.byteOffset >= image.initialBytes.size())
      return invalid("memory result has an invalid runtime offset");
    const llvm::ArrayRef<sim::SemanticMemoryByte> baseline(image.initialBytes);
    const llvm::ArrayRef<sim::SemanticMemoryByte> selectedBaseline =
        baseline.drop_front(plan.byteOffset);
    if (resultMemory.bytes.size() != selectedBaseline.size())
      return invalid("memory result byte count disagrees with the runtime "
                     "projection");
    if (plan.form == sim::MemoryObservationForm::FullState) {
      observations.memories.emplace_back(
          sim::FullMemoryObservation{resultMemory.bytes});
      continue;
    }

    sim::DiffMemoryObservation diff;
    diff.byteCount = resultMemory.bytes.size();
    std::size_t cursor = 0;
    while (cursor != resultMemory.bytes.size()) {
      const auto same = [&](std::size_t index) {
        const sim::SemanticMemoryByte &lhs = resultMemory.bytes[index];
        const sim::SemanticMemoryByte &rhs = selectedBaseline[index];
        return lhs.state == rhs.state &&
               (lhs.state != sim::SemanticState::Defined ||
                lhs.value == rhs.value);
      };
      if (same(cursor)) {
        ++cursor;
        continue;
      }
      const std::size_t begin = cursor;
      while (cursor != resultMemory.bytes.size() && !same(cursor))
        ++cursor;
      diff.runs.push_back(
          sim::MemoryDiffRun{begin, std::vector<sim::SemanticMemoryByte>(
                                        resultMemory.bytes.begin() + begin,
                                        resultMemory.bytes.begin() + cursor)});
    }
    observations.memories.emplace_back(std::move(diff));
  }
  return observations;
}

} // namespace loom::eda::open_source::detail
