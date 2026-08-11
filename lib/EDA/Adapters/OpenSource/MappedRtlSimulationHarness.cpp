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
#include <set>
#include <string>
#include <system_error>
#include <utility>

namespace loom::eda::open_source::detail {
namespace {

constexpr unsigned kBitsPerByte = 8;
constexpr unsigned kBitsPerHexDigit = 4;
constexpr unsigned kAxiResponseWidth = 2;
constexpr unsigned kConfigurationHandshakeCycleLimit = 64;
constexpr unsigned kResetReleaseCycleCount = 4;
constexpr std::uint64_t kMinimumClockPeriodFs = 2;
constexpr std::uint64_t kMaximumMemoryAddressWidth =
    std::numeric_limits<std::uint64_t>::digits;

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
           << "  longint unsigned loom_memory_byte_address_" << ordinal << ";\n"
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

  output << "  always_ff @(posedge " << facts.selectedClock << ") begin\n"
         << "    if (!loom_resets_released) begin\n";
  for (const auto &[ordinal, port] :
       llvm::enumerate(facts.memoryBoundaryPorts)) {
    (void)port;
    output << "      loom_memory_response_pending_" << ordinal << " <= 0;\n"
           << "      loom_memory_response_data_" << ordinal << " <= '0;\n";
  }
  output << "    end else begin\n";
  for (const auto &[ordinal, port] :
       llvm::enumerate(facts.memoryBoundaryPorts)) {
    const RuntimeMemoryImage &root = facts.memoryImages[port.rootObjectOrdinal];
    const std::uint64_t rootBase =
        root.canonicalBaseAddress + port.rootByteOffset;
    const std::uint64_t dataBytes = port.dataBitWidth / kBitsPerByte;
    output << "      if (loom_memory_response_pending_" << ordinal << " && "
           << port.prefix << "_response_ready) "
           << "loom_memory_response_pending_" << ordinal << " <= 0;\n"
           << "      if (" << port.prefix << "_request_valid && " << port.prefix
           << "_request_ready) begin\n"
           << "        loom_memory_response_next_" << ordinal << " = '0;\n"
           << "        if (" << port.prefix << "_request_element_width == 0 || "
           << port.prefix << "_request_element_width[2:0] != 0 || "
           << port.prefix << "_request_element_width > " << port.dataBitWidth
           << ") $fatal(1, \"unsupported external memory element width\");\n"
           << "        if (" << port.prefix
           << "_request_address_lane_width == 0 || " << port.prefix
           << "_request_address_lane_width > " << kMaximumMemoryAddressWidth
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
           << "_request_address_lane_width < " << kMaximumMemoryAddressWidth
           << ") loom_memory_lane_address_" << ordinal
           << " = loom_memory_lane_address_" << ordinal << " & (("
           << kMaximumMemoryAddressWidth << "'h1 << " << port.prefix
           << "_request_address_lane_width) - 1);\n"
           << "            if (" << port.prefix
           << "_request_address_form == 0) loom_memory_byte_address_" << ordinal
           << " = " << rootBase << " + " << port.prefix
           << "_request_base_address + loom_memory_lane_address_" << ordinal
           << " * loom_memory_element_bytes_" << ordinal << " + (("
           << port.prefix << "_request_access_form == 2) ? "
           << "loom_memory_byte_in_lane_" << ordinal << " : "
           << "loom_memory_byte_ordinal_" << ordinal << ");\n"
           << "            else loom_memory_byte_address_" << ordinal
           << " = loom_memory_lane_address_" << ordinal << " + (("
           << port.prefix << "_request_access_form == 2) ? "
           << "loom_memory_byte_in_lane_" << ordinal << " : "
           << "loom_memory_byte_ordinal_" << ordinal << ");\n"
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
           << "        if (loom_debug_verbose >= 2) $display("
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

void renderConfigurationTask(llvm::raw_ostream &output, llvm::StringRef prefix,
                             std::size_t ordinal, llvm::StringRef clock) {
  output << "  task automatic loom_cfg_write_" << ordinal << "(input logic ["
         << hardware::rtl::portableConfigurationAddressWidth - 1
         << ":0] address, input logic ["
         << hardware::rtl::portableConfigurationDataWidth - 1
         << ":0] data, input logic ["
         << hardware::rtl::portableConfigurationByteCount - 1
         << ":0] strobe);\n"
         << "    integer wait_cycles;\n"
         << "    begin\n"
         << "      if (loom_debug_verbose >= 1) $display(\"[loom][rtl][cfg] "
            "write address=%h data=%h strobe=%h\", address, data, strobe);\n"
         << "      " << prefix << "_awaddr = address;\n"
         << "      " << prefix << "_awvalid = 1;\n"
         << "      wait_cycles = 0;\n"
         << "      do begin\n"
         << "        @(posedge " << clock << ");\n"
         << "        wait_cycles = wait_cycles + 1;\n"
         << "        if (wait_cycles == " << kConfigurationHandshakeCycleLimit
         << " && !" << prefix
         << "_awready) $fatal(1, \"AXI4-Lite AW handshake timed out\");\n"
         << "      end while (!" << prefix << "_awready);\n"
         << "      #1 " << prefix << "_awvalid = 0;\n"
         << "      " << prefix << "_wdata = data;\n"
         << "      " << prefix << "_wstrb = strobe;\n"
         << "      " << prefix << "_wvalid = 1;\n"
         << "      wait_cycles = 0;\n"
         << "      do begin\n"
         << "        @(posedge " << clock << ");\n"
         << "        wait_cycles = wait_cycles + 1;\n"
         << "        if (wait_cycles == " << kConfigurationHandshakeCycleLimit
         << " && !" << prefix
         << "_wready) $fatal(1, \"AXI4-Lite W handshake timed out\");\n"
         << "      end while (!" << prefix << "_wready);\n"
         << "      #1 " << prefix << "_wvalid = 0;\n"
         << "      wait_cycles = 0;\n"
         << "      do begin\n"
         << "        @(negedge " << clock << ");\n"
         << "        wait_cycles = wait_cycles + 1;\n"
         << "        if (wait_cycles == " << kConfigurationHandshakeCycleLimit
         << " && !" << prefix
         << "_bvalid) $fatal(1, \"AXI4-Lite B response timed out\");\n"
         << "      end while (!" << prefix << "_bvalid);\n"
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
         << "      " << prefix << "_araddr = address;\n"
         << "      " << prefix << "_arvalid = 1;\n"
         << "      wait_cycles = 0;\n"
         << "      do begin\n"
         << "        @(posedge " << clock << ");\n"
         << "        wait_cycles = wait_cycles + 1;\n"
         << "        if (wait_cycles == " << kConfigurationHandshakeCycleLimit
         << " && !" << prefix
         << "_arready) $fatal(1, \"AXI4-Lite AR handshake timed out\");\n"
         << "      end while (!" << prefix << "_arready);\n"
         << "      #1 " << prefix << "_arvalid = 0;\n"
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
  for (std::uint64_t word = 0; word != program.layout.payloadWordCount;
       ++word) {
    const std::uint32_t address =
        program.layout.baseAddress +
        static_cast<std::uint32_t>(
            word * hardware::rtl::portableConfigurationByteCount);
    output << "    loom_cfg_write_" << taskOrdinal << "("
           << hardware::rtl::portableConfigurationAddressWidth << "'h"
           << llvm::format_hex_no_prefix(
                  address, hardware::rtl::portableConfigurationAddressWidth /
                               kBitsPerHexDigit)
           << ", " << hardware::rtl::portableConfigurationDataWidth << "'h"
           << llvm::format_hex_no_prefix(
                  imageWord(program.image, word),
                  hardware::rtl::portableConfigurationDataWidth /
                      kBitsPerHexDigit)
           << ", " << hardware::rtl::portableConfigurationByteCount << "'h"
           << llvm::format_hex_no_prefix(imageStrobe(program.image, word), 1)
           << ");\n";
  }
  output << "    loom_cfg_write_" << taskOrdinal << "("
         << hardware::rtl::portableConfigurationAddressWidth << "'h"
         << llvm::format_hex_no_prefix(
                program.layout.commitAddress,
                hardware::rtl::portableConfigurationAddressWidth /
                    kBitsPerHexDigit)
         << ", " << hardware::rtl::portableConfigurationDataWidth
         << "'h00000001, " << hardware::rtl::portableConfigurationByteCount
         << "'h1);\n";
  for (std::uint64_t word = 0; word != program.layout.payloadWordCount;
       ++word) {
    const std::uint32_t address =
        program.layout.baseAddress +
        static_cast<std::uint32_t>(
            word * hardware::rtl::portableConfigurationByteCount);
    output << "    loom_cfg_read_" << taskOrdinal << "("
           << hardware::rtl::portableConfigurationAddressWidth << "'h"
           << llvm::format_hex_no_prefix(
                  address, hardware::rtl::portableConfigurationAddressWidth /
                               kBitsPerHexDigit)
           << ", loom_cfg_readback, loom_cfg_response);\n"
           << "    if (loom_cfg_response !== 2'b00 || "
              "loom_cfg_readback !== "
           << hardware::rtl::portableConfigurationDataWidth << "'h"
           << llvm::format_hex_no_prefix(
                  imageWord(program.image, word),
                  hardware::rtl::portableConfigurationDataWidth /
                      kBitsPerHexDigit)
           << ") $fatal(1, \"active configuration readback mismatch\");\n";
  }
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
            "\"configuration status did not clear\");\n";
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
renderMappedRtlTestbench(const MappedRtlInvocationFacts &facts,
                         llvm::StringRef resultPath) {
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
  auto projectedMemoryExtent = memoryExtent(facts);
  if (!projectedMemoryExtent)
    return projectedMemoryExtent.takeError();
  for (const MemoryBoundaryPort &port : facts.memoryBoundaryPorts) {
    if (port.addressBitWidth == 0 ||
        port.addressBitWidth > kMaximumMemoryAddressWidth ||
        port.dataBitWidth == 0 || port.dataBitWidth % kBitsPerByte != 0 ||
        port.maskBitWidth == 0 ||
        port.rootObjectOrdinal >= facts.memoryImages.size() ||
        port.rootByteOffset >=
            facts.memoryImages[port.rootObjectOrdinal].initialBytes.size())
      return invalid("memory boundary plan is incomplete");
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
         << "  longint unsigned loom_cycle;\n"
         << "  longint unsigned loom_launch_cycle;\n"
         << "  longint unsigned loom_retirement_cycle;\n"
         << "  logic [" << hardware::rtl::portableConfigurationDataWidth - 1
         << ":0] loom_cfg_readback;\n"
         << "  logic [" << kAxiResponseWidth - 1 << ":0] loom_cfg_response;\n"
         << "  integer loom_debug_verbose;\n";
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
         << "    if (!$value$plusargs(\"LOOM_DEBUG_VERBOSE=%d\", "
            "loom_debug_verbose)) loom_debug_verbose = 0;\n";
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
  output << "    repeat (" << kResetReleaseCycleCount << ") @(posedge "
         << facts.selectedClock << ");\n"
         << "    @(negedge " << facts.selectedClock << ");\n";
  for (const ResetPort &reset : facts.resetPorts)
    output << "    " << reset.name << " = " << !reset.assertedValue << ";\n";
  output << "    loom_resets_released = 1;\n";
  for (const auto &[ordinal, program] :
       llvm::enumerate(facts.configurationPrograms))
    renderConfigurationProgram(output, program, ordinal);
  output << "    loom_inputs_enabled = 1;\n"
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
           << "    if (loom_inputs_enabled && loom_debug_verbose >= 3) "
              "$display(\"[loom][rtl][input] ordinal="
           << ordinal << " cycle=%0d valid=%0d ready=%0d index=%0d\", "
           << "loom_cycle, " << input->port.prefix << "_valid, "
           << input->port.prefix << "_ready, loom_input_index_" << ordinal
           << ");\n"
           << "    if (loom_inputs_enabled && loom_debug_verbose >= 2 && "
           << input->port.prefix << "_valid && " << input->port.prefix
           << "_ready) $display(\"[loom][rtl][input] ordinal=" << ordinal
           << " accepted_cycle=%0d\", loom_cycle);\n"
           << "  end\n";
  }

  for (const auto &[ordinal, program] :
       llvm::enumerate(facts.configurationPrograms))
    renderConfigurationTask(output, program.portPrefix, ordinal,
                            facts.selectedClock);
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
         << "      if (loom_debug_verbose >= 3) $display("
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
         << "        if (loom_debug_verbose >= 1) $display("
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
    const MappedRtlInvocationFacts &facts, std::uint64_t buildJobs,
    llvm::StringRef testbenchPath, llvm::StringRef simulatorExecutablePath) {
  if (facts.rtlPaths.empty())
    return invalid("Verilator driver has no RTL sources");
  if (buildJobs == 0)
    return invalid("Verilator build parallelism must be positive");
  std::string text;
  llvm::raw_string_ostream output(text);
  const std::filesystem::path simulatorExecutable(
      simulatorExecutablePath.str());
  output << "--binary\n--build-jobs\n"
         << buildJobs
         << "\n--timing\n--Wall\n--Wno-fatal\n"
            "--Wno-DECLFILENAME\n--Wno-UNUSEDSIGNAL\n--Wno-PINMISSING\n"
            "--Wno-TIMESCALEMOD\n"
            "-CFLAGS\n-std=c++20\n--top-module\n"
         << mappedRtlHarnessTop << "\n--Mdir\n"
         << simulatorExecutable.parent_path().generic_string() << "\n-o\n"
         << simulatorExecutable.filename().generic_string() << "\n";
  for (const std::string &path : facts.rtlPaths)
    output << path << "\n";
  output << testbenchPath << "\n";
  return text;
}

llvm::Expected<std::string> renderMappedRtlBridgedVerilatorDriver(
    const MappedRtlInvocationFacts &facts, std::uint64_t buildJobs,
    llvm::StringRef testbenchPath, llvm::StringRef bridgeEngineSourcePath,
    llvm::StringRef simulatorExecutablePath) {
  if (facts.rtlPaths.empty())
    return invalid("Verilator driver has no RTL sources");
  if (buildJobs == 0)
    return invalid("Verilator build parallelism must be positive");
  std::string text;
  llvm::raw_string_ostream output(text);
  const std::filesystem::path simulatorExecutable(
      simulatorExecutablePath.str());
  output << "--cc\n--exe\n--build\n--build-jobs\n"
         << buildJobs
         << "\n--timing\n--Wall\n--Wno-fatal\n"
            "--Wno-DECLFILENAME\n--Wno-UNUSEDSIGNAL\n--Wno-PINMISSING\n"
            "--Wno-TIMESCALEMOD\n"
            "-CFLAGS\n-std=c++20\n--top-module\n"
         << mappedRtlHarnessTop << "\n--Mdir\n"
         << simulatorExecutable.parent_path().generic_string() << "\n-o\n"
         << simulatorExecutable.filename().generic_string() << "\n";
  for (const std::string &path : facts.rtlPaths)
    output << path << "\n";
  output << testbenchPath << "\n" << bridgeEngineSourcePath << "\n";
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
