//===-- TemporalDomainADGs.cpp - Temporal e2e ADG builders -----*- C++ -*-===//
//
// Part of the fcc project.
//
//===----------------------------------------------------------------------===//

#include "TemporalDomainADGs.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/Twine.h"

#include <memory>
#include <sstream>
#include <system_error>

namespace fcc {
namespace e2e {

static std::string bitsType(unsigned width) {
  return "!fabric.bits<" + std::to_string(width) + ">";
}

static std::string taggedBitsType(unsigned width) {
  return "!fabric.tagged<" + bitsType(width) + ", i1>";
}

static std::unique_ptr<llvm::raw_fd_ostream>
openOutput(const std::string &outputPath) {
  std::error_code ec;
  auto os =
      std::make_unique<llvm::raw_fd_ostream>(outputPath, ec,
                                             llvm::sys::fs::OF_Text);
  if (ec)
    llvm::report_fatal_error(llvm::Twine("cannot open output file: ") +
                             outputPath);
  return os;
}

void buildTemporalReductionDomain(const std::string &outputPath,
                                  const TemporalReductionDomainOptions &opts) {
  auto os = openOutput(outputPath);

  const std::string bitsTy = bitsType(opts.dataWidth);
  const std::string taggedTy = taggedBitsType(opts.dataWidth);

  *os << "module {\n";
  *os << "  fabric.temporal_pe @tpe_reduce(\n";
  *os << "      %p0: " << taggedTy << ",\n";
  *os << "      %p1: " << taggedTy << ",\n";
  *os << "      %p2: " << taggedTy << ",\n";
  *os << "      %p3: " << taggedTy << ",\n";
  *os << "      %p4: " << taggedTy << ")\n";
  *os << "      -> (" << taggedTy << ",\n";
  *os << "          " << taggedTy << ",\n";
  *os << "          " << taggedTy << ")\n";
  *os << "      [\n";
  *os << "        num_register = " << opts.numRegister << " : i64,\n";
  *os << "        num_instruction = " << opts.numInstruction << " : i64,\n";
  *os << "        reg_fifo_depth = " << opts.regFifoDepth << " : i64\n";
  *os << "      ] {\n";
  *os << "    fabric.function_unit @fu_join(%a: none) -> (none)\n";
  *os << "        [latency = 1, interval = 1] {\n";
  *os << "      %0 = handshake.join %a : none\n";
  *os << "      fabric.yield %0 : none\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_join_1(%a: none, %b: none) -> (none)\n";
  *os << "        [latency = 1, interval = 1] {\n";
  *os << "      %0 = handshake.join %a, %b : none, none\n";
  *os << "      fabric.yield %0 : none\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_const_index(%ctrl: none) -> (index)\n";
  *os << "        [latency = 1, interval = 1] {\n";
  *os << "      %0 = handshake.constant %ctrl {value = 0 : index} : index\n";
  *os << "      fabric.yield %0 : index\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_const_index_1(%ctrl: none) -> (index)\n";
  *os << "        [latency = 1, interval = 1] {\n";
  *os << "      %0 = handshake.constant %ctrl {value = 0 : index} : index\n";
  *os << "      fabric.yield %0 : index\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_const_i32(%ctrl: none) -> (i32)\n";
  *os << "        [latency = 1, interval = 1] {\n";
  *os << "      %0 = handshake.constant %ctrl {value = 0 : i32} : i32\n";
  *os << "      fabric.yield %0 : i32\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_index_cast(%arg0: i32) -> (index)\n";
  *os << "        [latency = 1, interval = 1] {\n";
  *os << "      %0 = arith.index_cast %arg0 : i32 to index\n";
  *os << "      fabric.yield %0 : index\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_stream(%start: index, %step: index, %bound: index)\n";
  *os << "        -> (index, i1) [latency = 1, interval = 1] {\n";
  *os << "      %0, %1 = dataflow.stream %start, %step, %bound\n";
  *os << "          {step_op = \"+=\", cont_cond = \"<\"}\n";
  *os << "          : (index, index, index) -> (index, i1)\n";
  *os << "      fabric.yield %0, %1 : index, i1\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_gate_index(%value: index, %cond: i1)\n";
  *os << "        -> (index, i1) [latency = 1, interval = 1] {\n";
  *os << "      %0, %1 = dataflow.gate %value, %cond : index, i1 -> index, i1\n";
  *os << "      fabric.yield %0, %1 : index, i1\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_gate_i32(%value: i32, %cond: i1)\n";
  *os << "        -> (i32, i1) [latency = 1, interval = 1] {\n";
  *os << "      %0, %1 = dataflow.gate %value, %cond : i32, i1 -> i32, i1\n";
  *os << "      fabric.yield %0, %1 : i32, i1\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_carry_i32(%cond: i1, %a: i32, %b: i32)\n";
  *os << "        -> (i32) [latency = 1, interval = 1] {\n";
  *os << "      %0 = dataflow.carry %cond, %a, %b : i1, i32, i32 -> i32\n";
  *os << "      fabric.yield %0 : i32\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_carry_none(%cond: i1, %a: none, %b: none)\n";
  *os << "        -> (none) [latency = 1, interval = 1] {\n";
  *os << "      %0 = dataflow.carry %cond, %a, %b : i1, none, none -> none\n";
  *os << "      fabric.yield %0 : none\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_carry_none_1(%cond: i1, %a: none, %b: none)\n";
  *os << "        -> (none) [latency = 1, interval = 1] {\n";
  *os << "      %0 = dataflow.carry %cond, %a, %b : i1, none, none -> none\n";
  *os << "      fabric.yield %0 : none\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_cond_br_i32(%cond: i1, %value: i32)\n";
  *os << "        -> (i32, i32) [latency = 1, interval = 1] {\n";
  *os << "      %0, %1 = handshake.cond_br %cond, %value : i32\n";
  *os << "      fabric.yield %0, %1 : i32, i32\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_cond_br_none(%cond: i1, %value: none)\n";
  *os << "        -> (none, none) [latency = 1, interval = 1] {\n";
  *os << "      %0, %1 = handshake.cond_br %cond, %value : none\n";
  *os << "      fabric.yield %0, %1 : none, none\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_cond_br_none_1(%cond: i1, %value: none)\n";
  *os << "        -> (none, none) [latency = 1, interval = 1] {\n";
  *os << "      %0, %1 = handshake.cond_br %cond, %value : none\n";
  *os << "      fabric.yield %0, %1 : none, none\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_load(%addr: index, %data: i32, %ctrl: none)\n";
  *os << "        -> (i32, index) [latency = 1, interval = 1] {\n";
  *os << "      %0, %1 = handshake.load [%addr] %data, %ctrl : index, i32\n";
  *os << "      fabric.yield %0, %1 : i32, index\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_addi(%a: i32, %b: i32) -> (i32)\n";
  *os << "        [latency = 1, interval = 1] {\n";
  *os << "      %0 = arith.addi %a, %b : i32\n";
  *os << "      fabric.yield %0 : i32\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_cmpi(%a: i32, %b: i32) -> (i1)\n";
  *os << "        [latency = 1, interval = 1] {\n";
  *os << "      %0 = arith.cmpi eq, %a, %b : i32\n";
  *os << "      fabric.yield %0 : i1\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_select_index(%cond: i1, %a: index, %b: index)\n";
  *os << "        -> (index) [latency = 1, interval = 1] {\n";
  *os << "      %0 = arith.select %cond, %a, %b : index\n";
  *os << "      fabric.yield %0 : index\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_mux_i32(%sel: index, %a: i32, %b: i32)\n";
  *os << "        -> (i32) [latency = 1, interval = 1] {\n";
  *os << "      %0 = handshake.mux %sel [%a, %b] : index, i32\n";
  *os << "      fabric.yield %0 : i32\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_mux_none(%sel: index, %a: none, %b: none)\n";
  *os << "        -> (none) [latency = 1, interval = 1] {\n";
  *os << "      %0 = handshake.mux %sel [%a, %b] : index, none\n";
  *os << "      fabric.yield %0 : none\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_mux_none_1(%sel: index, %a: none, %b: none)\n";
  *os << "        -> (none) [latency = 1, interval = 1] {\n";
  *os << "      %0 = handshake.mux %sel [%a, %b] : index, none\n";
  *os << "      fabric.yield %0 : none\n";
  *os << "    }\n\n";
  *os << "    fabric.yield\n";
  *os << "  }\n\n";
  *os << "  fabric.module @" << opts.moduleName << "(\n";
  *os << "      %mem0: " << opts.memrefType << ",\n";
  *os << "      %n: " << bitsTy << ",\n";
  *os << "      %init: " << bitsTy << ",\n";
  *os << "      %ctrl: " << bitsTy << ")\n";
  *os << "      -> (" << bitsTy << ", " << bitsTy << ") {\n";
  *os << "    %ext0:2 = fabric.extmemory @extmem_0\n";
  *os << "        [ldCount = 1, stCount = 0, lsqDepth = 0, memrefType = "
     << opts.memrefType << "]\n";
  *os << "        (%mem0, %addr_bits)\n";
  *os << "        : (" << opts.memrefType << ", " << bitsTy << ")\n";
  *os << "          -> (" << bitsTy << ", " << bitsTy << ")\n\n";
  *os << "    %tag_n = fabric.add_tag %n {tag = 0 : i64}\n";
  *os << "        : " << bitsTy << " -> " << taggedTy << "\n";
  *os << "    %tag_init = fabric.add_tag %init {tag = 0 : i64}\n";
  *os << "        : " << bitsTy << " -> " << taggedTy << "\n";
  *os << "    %tag_ctrl = fabric.add_tag %ctrl {tag = 0 : i64}\n";
  *os << "        : " << bitsTy << " -> " << taggedTy << "\n";
  *os << "    %tag_lddata = fabric.add_tag %ext0#0 {tag = 0 : i64}\n";
  *os << "        : " << bitsTy << " -> " << taggedTy << "\n";
  *os << "    %tag_lddone = fabric.add_tag %ext0#1 {tag = 0 : i64}\n";
  *os << "        : " << bitsTy << " -> " << taggedTy << "\n\n";
  *os << "    %tpe0:3 = fabric.instance @tpe_reduce(\n";
  *os << "        %tag_n, %tag_init, %tag_ctrl, %tag_lddata, %tag_lddone)\n";
  *os << "        {sym_name = \"tpe_0\"}\n";
  *os << "        : (" << taggedTy << ",\n";
  *os << "           " << taggedTy << ",\n";
  *os << "           " << taggedTy << ",\n";
  *os << "           " << taggedTy << ",\n";
  *os << "           " << taggedTy << ")\n";
  *os << "          -> (" << taggedTy << ",\n";
  *os << "              " << taggedTy << ",\n";
  *os << "              " << taggedTy << ")\n\n";
  *os << "    %addr_bits = fabric.del_tag %tpe0#0\n";
  *os << "        : " << taggedTy << " -> " << bitsTy << "\n";
  *os << "    %sum_bits = fabric.del_tag %tpe0#1\n";
  *os << "        : " << taggedTy << " -> " << bitsTy << "\n";
  *os << "    %done_bits = fabric.del_tag %tpe0#2\n";
  *os << "        : " << taggedTy << " -> " << bitsTy << "\n\n";
  *os << "    fabric.yield %sum_bits, %done_bits : " << bitsTy << ", " << bitsTy
     << "\n";
  *os << "  }\n";
  *os << "}\n";
}

void buildTemporalScanDomain(const std::string &outputPath,
                             const TemporalScanDomainOptions &opts) {
  auto os = openOutput(outputPath);

  const std::string bitsTy = bitsType(opts.dataWidth);
  const std::string taggedTy = taggedBitsType(opts.dataWidth);

  *os << "module {\n";
  *os << "  fabric.temporal_pe @tpe_scan(\n";
  *os << "      %p0: " << taggedTy << ",\n";
  *os << "      %p1: " << taggedTy << ",\n";
  *os << "      %p2: " << taggedTy << ",\n";
  *os << "      %p3: " << taggedTy << ",\n";
  *os << "      %p4: " << taggedTy << ")\n";
  *os << "      -> (" << taggedTy << ",\n";
  *os << "          " << taggedTy << ",\n";
  *os << "          " << taggedTy << ",\n";
  *os << "          " << taggedTy << ")\n";
  *os << "      [\n";
  *os << "        num_register = " << opts.numRegister << " : i64,\n";
  *os << "        num_instruction = " << opts.numInstruction << " : i64,\n";
  *os << "        reg_fifo_depth = " << opts.regFifoDepth << " : i64\n";
  *os << "      ] {\n";
  *os << "    fabric.function_unit @fu_join(%a: none) -> (none)\n";
  *os << "        [latency = 1, interval = 1] {\n";
  *os << "      %0 = handshake.join %a : none\n";
  *os << "      fabric.yield %0 : none\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_join_1(%a: none, %b: none) -> (none)\n";
  *os << "        [latency = 1, interval = 1] {\n";
  *os << "      %0 = handshake.join %a, %b : none, none\n";
  *os << "      fabric.yield %0 : none\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_const_index(%ctrl: none) -> (index)\n";
  *os << "        [latency = 1, interval = 1] {\n";
  *os << "      %0 = handshake.constant %ctrl {value = 0 : index} : index\n";
  *os << "      fabric.yield %0 : index\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_const_index_1(%ctrl: none) -> (index)\n";
  *os << "        [latency = 1, interval = 1] {\n";
  *os << "      %0 = handshake.constant %ctrl {value = 1 : index} : index\n";
  *os << "      fabric.yield %0 : index\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_const_i32(%ctrl: none) -> (i32)\n";
  *os << "        [latency = 1, interval = 1] {\n";
  *os << "      %0 = handshake.constant %ctrl {value = 0 : i32} : i32\n";
  *os << "      fabric.yield %0 : i32\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_index_cast(%arg0: i32) -> (index)\n";
  *os << "        [latency = 1, interval = 1] {\n";
  *os << "      %0 = arith.index_cast %arg0 : i32 to index\n";
  *os << "      fabric.yield %0 : index\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_stream(%start: index, %step: index, %bound: index)\n";
  *os << "        -> (index, i1) [latency = 1, interval = 1] {\n";
  *os << "      %0, %1 = dataflow.stream %start, %step, %bound\n";
  *os << "          {step_op = \"+=\", cont_cond = \"<\"}\n";
  *os << "          : (index, index, index) -> (index, i1)\n";
  *os << "      fabric.yield %0, %1 : index, i1\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_gate_index(%value: index, %cond: i1)\n";
  *os << "        -> (index, i1) [latency = 1, interval = 1] {\n";
  *os << "      %0, %1 = dataflow.gate %value, %cond : index, i1 -> index, i1\n";
  *os << "      fabric.yield %0, %1 : index, i1\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_gate_i32(%value: i32, %cond: i1)\n";
  *os << "        -> (i32, i1) [latency = 1, interval = 1] {\n";
  *os << "      %0, %1 = dataflow.gate %value, %cond : i32, i1 -> i32, i1\n";
  *os << "      fabric.yield %0, %1 : i32, i1\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_carry_i32(%cond: i1, %a: i32, %b: i32)\n";
  *os << "        -> (i32) [latency = 1, interval = 1] {\n";
  *os << "      %0 = dataflow.carry %cond, %a, %b : i1, i32, i32 -> i32\n";
  *os << "      fabric.yield %0 : i32\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_carry_none(%cond: i1, %a: none, %b: none)\n";
  *os << "        -> (none) [latency = 1, interval = 1] {\n";
  *os << "      %0 = dataflow.carry %cond, %a, %b : i1, none, none -> none\n";
  *os << "      fabric.yield %0 : none\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_carry_none_1(%cond: i1, %a: none, %b: none)\n";
  *os << "        -> (none) [latency = 1, interval = 1] {\n";
  *os << "      %0 = dataflow.carry %cond, %a, %b : i1, none, none -> none\n";
  *os << "      fabric.yield %0 : none\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_cond_br_none(%cond: i1, %value: none)\n";
  *os << "        -> (none, none) [latency = 1, interval = 1] {\n";
  *os << "      %0, %1 = handshake.cond_br %cond, %value : none\n";
  *os << "      fabric.yield %0, %1 : none, none\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_cond_br_none_1(%cond: i1, %value: none)\n";
  *os << "        -> (none, none) [latency = 1, interval = 1] {\n";
  *os << "      %0, %1 = handshake.cond_br %cond, %value : none\n";
  *os << "      fabric.yield %0, %1 : none, none\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_cond_br_none_2(%cond: i1, %value: none)\n";
  *os << "        -> (none, none) [latency = 1, interval = 1] {\n";
  *os << "      %0, %1 = handshake.cond_br %cond, %value : none\n";
  *os << "      fabric.yield %0, %1 : none, none\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_load(%addr: index, %data: i32, %ctrl: none)\n";
  *os << "        -> (i32, index) [latency = 1, interval = 1] {\n";
  *os << "      %0, %1 = handshake.load [%addr] %data, %ctrl : index, i32\n";
  *os << "      fabric.yield %0, %1 : i32, index\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_store(%addr: index, %data: i32, %ctrl: none)\n";
  *os << "        -> (i32, index) [latency = 1, interval = 1] {\n";
  *os << "      %0, %1 = handshake.store [%addr] %data, %ctrl : index, i32\n";
  *os << "      fabric.yield %0, %1 : i32, index\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_addi(%a: i32, %b: i32) -> (i32)\n";
  *os << "        [latency = 1, interval = 1] {\n";
  *os << "      %0 = arith.addi %a, %b : i32\n";
  *os << "      fabric.yield %0 : i32\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_cmpi(%a: i32, %b: i32) -> (i1)\n";
  *os << "        [latency = 1, interval = 1] {\n";
  *os << "      %0 = arith.cmpi eq, %a, %b : i32\n";
  *os << "      fabric.yield %0 : i1\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_select_index(%cond: i1, %a: index, %b: index)\n";
  *os << "        -> (index) [latency = 1, interval = 1] {\n";
  *os << "      %0 = arith.select %cond, %a, %b : index\n";
  *os << "      fabric.yield %0 : index\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_mux_none(%sel: index, %a: none, %b: none)\n";
  *os << "        -> (none) [latency = 1, interval = 1] {\n";
  *os << "      %0 = handshake.mux %sel [%a, %b] : index, none\n";
  *os << "      fabric.yield %0 : none\n";
  *os << "    }\n\n";
  *os << "    fabric.function_unit @fu_mux_none_1(%sel: index, %a: none, %b: none)\n";
  *os << "        -> (none) [latency = 1, interval = 1] {\n";
  *os << "      %0 = handshake.mux %sel [%a, %b] : index, none\n";
  *os << "      fabric.yield %0 : none\n";
  *os << "    }\n\n";
  *os << "    fabric.yield\n";
  *os << "  }\n\n";
  *os << "  fabric.module @" << opts.moduleName << "(\n";
  *os << "      %mem_in: " << opts.memrefType << ",\n";
  *os << "      %mem_out: " << opts.memrefType << ",\n";
  *os << "      %n: " << bitsTy << ",\n";
  *os << "      %ctrl: " << bitsTy << ")\n";
  *os << "      -> (" << bitsTy << ") {\n";
  *os << "    %ld0:2 = fabric.extmemory @extmem_in\n";
  *os << "        [ldCount = 1, stCount = 0, lsqDepth = 0, memrefType = "
     << opts.memrefType << "]\n";
  *os << "        (%mem_in, %ld_addr_bits)\n";
  *os << "        : (" << opts.memrefType << ", " << bitsTy << ")\n";
  *os << "          -> (" << bitsTy << ", " << bitsTy << ")\n\n";
  *os << "    %st0 = fabric.extmemory @extmem_out\n";
  *os << "        [ldCount = 0, stCount = 1, lsqDepth = 0, memrefType = "
     << opts.memrefType << "]\n";
  *os << "        (%mem_out, %st_addr_bits, %st_data_bits)\n";
  *os << "        : (" << opts.memrefType << ", " << bitsTy << ", " << bitsTy
     << ")\n";
  *os << "          -> (" << bitsTy << ")\n\n";
  *os << "    %tag_n = fabric.add_tag %n {tag = 0 : i64}\n";
  *os << "        : " << bitsTy << " -> " << taggedTy << "\n";
  *os << "    %tag_ctrl = fabric.add_tag %ctrl {tag = 0 : i64}\n";
  *os << "        : " << bitsTy << " -> " << taggedTy << "\n";
  *os << "    %tag_lddata = fabric.add_tag %ld0#0 {tag = 0 : i64}\n";
  *os << "        : " << bitsTy << " -> " << taggedTy << "\n";
  *os << "    %tag_lddone = fabric.add_tag %ld0#1 {tag = 0 : i64}\n";
  *os << "        : " << bitsTy << " -> " << taggedTy << "\n";
  *os << "    %tag_stdone = fabric.add_tag %st0 {tag = 0 : i64}\n";
  *os << "        : " << bitsTy << " -> " << taggedTy << "\n\n";
  *os << "    %tpe0:4 = fabric.instance @tpe_scan(\n";
  *os << "        %tag_n, %tag_ctrl, %tag_lddata, %tag_lddone, %tag_stdone)\n";
  *os << "        {sym_name = \"tpe_0\"}\n";
  *os << "        : (" << taggedTy << ",\n";
  *os << "           " << taggedTy << ",\n";
  *os << "           " << taggedTy << ",\n";
  *os << "           " << taggedTy << ",\n";
  *os << "           " << taggedTy << ")\n";
  *os << "          -> (" << taggedTy << ",\n";
  *os << "              " << taggedTy << ",\n";
  *os << "              " << taggedTy << ",\n";
  *os << "              " << taggedTy << ")\n\n";
  *os << "    %ld_addr_bits = fabric.del_tag %tpe0#0\n";
  *os << "        : " << taggedTy << " -> " << bitsTy << "\n";
  *os << "    %st_data_bits = fabric.del_tag %tpe0#1\n";
  *os << "        : " << taggedTy << " -> " << bitsTy << "\n";
  *os << "    %st_addr_bits = fabric.del_tag %tpe0#2\n";
  *os << "        : " << taggedTy << " -> " << bitsTy << "\n";
  *os << "    %done_bits = fabric.del_tag %tpe0#3\n";
  *os << "        : " << taggedTy << " -> " << bitsTy << "\n\n";
  *os << "    fabric.yield %done_bits : " << bitsTy << "\n";
  *os << "  }\n";
  *os << "}\n";
}

} // namespace e2e
} // namespace fcc
