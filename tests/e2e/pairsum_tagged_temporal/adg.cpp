//===-- adg.cpp - pairsum_tagged_temporal ADG generator --------*- C++ -*-===//
//
// Part of the fcc project.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

static llvm::cl::opt<std::string> outputFile(
    llvm::cl::Positional,
    llvm::cl::desc("<output .fabric.mlir file>"),
    llvm::cl::init("pairsum_tagged_temporal.fabric.mlir"));

namespace {

constexpr const char *kPairsumTaggedTemporalADG = R"FCCADG(module {
  fabric.spatial_sw @addr_mux
      [connectivity_table = ["11"]]
      : (!fabric.tagged<!fabric.bits<64>, i1>,
         !fabric.tagged<!fabric.bits<64>, i1>)
        -> (!fabric.tagged<!fabric.bits<57>, i1>)

  fabric.temporal_sw @ld_data_demux
      [num_route_table = 2, connectivity_table = ["1", "1"]]
      : (!fabric.tagged<!fabric.bits<64>, i1>)
        -> (!fabric.tagged<!fabric.bits<64>, i1>,
            !fabric.tagged<!fabric.bits<64>, i1>)

  fabric.temporal_sw @ld_done_demux
      [num_route_table = 2, connectivity_table = ["1", "1"]]
      : (!fabric.tagged<!fabric.bits<64>, i1>)
        -> (!fabric.tagged<!fabric.bits<64>, i1>,
            !fabric.tagged<!fabric.bits<64>, i1>)

  fabric.temporal_pe @tpe_pairsum(
      %p0: !fabric.tagged<!fabric.bits<64>, i1>,
      %p1: !fabric.tagged<!fabric.bits<64>, i1>,
      %p2: !fabric.tagged<!fabric.bits<64>, i1>,
      %p3: !fabric.tagged<!fabric.bits<64>, i1>,
      %p4: !fabric.tagged<!fabric.bits<64>, i1>,
      %p5: !fabric.tagged<!fabric.bits<64>, i1>)
      -> (!fabric.tagged<!fabric.bits<64>, i1>,
          !fabric.tagged<!fabric.bits<64>, i1>,
          !fabric.tagged<!fabric.bits<64>, i1>,
          !fabric.tagged<!fabric.bits<64>, i1>)
      [
        num_register = 0 : i64,
        num_instruction = 24 : i64,
        reg_fifo_depth = 0 : i64
      ] {
    fabric.function_unit @fu_join(%a: none) -> (none)
        [latency = 1, interval = 1] {
      %0 = handshake.join %a : none
      fabric.yield %0 : none
    }

    fabric.function_unit @fu_const_index(%ctrl: none) -> (index)
        [latency = 1, interval = 1] {
      %0 = handshake.constant %ctrl {value = 0 : index} : index
      fabric.yield %0 : index
    }

    fabric.function_unit @fu_const_i32(%ctrl: none) -> (i32)
        [latency = 1, interval = 1] {
      %0 = handshake.constant %ctrl {value = 0 : i32} : i32
      fabric.yield %0 : i32
    }

    fabric.function_unit @fu_const_i32_1(%ctrl: none) -> (i32)
        [latency = 1, interval = 1] {
      %0 = handshake.constant %ctrl {value = 0 : i32} : i32
      fabric.yield %0 : i32
    }

    fabric.function_unit @fu_const_i32_2(%ctrl: none) -> (i32)
        [latency = 1, interval = 1] {
      %0 = handshake.constant %ctrl {value = 0 : i32} : i32
      fabric.yield %0 : i32
    }

    fabric.function_unit @fu_const_index_1(%ctrl: none) -> (index)
        [latency = 1, interval = 1] {
      %0 = handshake.constant %ctrl {value = 0 : index} : index
      fabric.yield %0 : index
    }

    fabric.function_unit @fu_const_i64(%ctrl: none) -> (i64)
        [latency = 1, interval = 1] {
      %0 = handshake.constant %ctrl {value = 0 : i64} : i64
      fabric.yield %0 : i64
    }

    fabric.function_unit @fu_stream(%start: index, %step: index, %bound: index)
        -> (index, i1) [latency = 1, interval = 1] {
      %0, %1 = dataflow.stream %start, %step, %bound {step_op = "+=", cont_cond = "<"}
          : (index, index, index) -> (index, i1)
      fabric.yield %0, %1 : index, i1
    }

    fabric.function_unit @fu_gate_index(%value: index, %cond: i1)
        -> (index, i1) [latency = 1, interval = 1] {
      %0, %1 = dataflow.gate %value, %cond : index, i1 -> index, i1
      fabric.yield %0, %1 : index, i1
    }

    fabric.function_unit @fu_gate_i32(%value: i32, %cond: i1)
        -> (i32, i1) [latency = 1, interval = 1] {
      %0, %1 = dataflow.gate %value, %cond : i32, i1 -> i32, i1
      fabric.yield %0, %1 : i32, i1
    }

    fabric.function_unit @fu_gate_i64(%value: i64, %cond: i1)
        -> (i64, i1) [latency = 1, interval = 1] {
      %0, %1 = dataflow.gate %value, %cond : i64, i1 -> i64, i1
      fabric.yield %0, %1 : i64, i1
    }

    fabric.function_unit @fu_carry_i32(%cond: i1, %a: i32, %b: i32)
        -> (i32) [latency = 1, interval = 1] {
      %0 = dataflow.carry %cond, %a, %b : i1, i32, i32 -> i32
      fabric.yield %0 : i32
    }

    fabric.function_unit @fu_carry_i64(%cond: i1, %a: i64, %b: i64)
        -> (i64) [latency = 1, interval = 1] {
      %0 = dataflow.carry %cond, %a, %b : i1, i64, i64 -> i64
      fabric.yield %0 : i64
    }

    fabric.function_unit @fu_carry_none(%cond: i1, %a: none, %b: none)
        -> (none) [latency = 1, interval = 1] {
      %0 = dataflow.carry %cond, %a, %b : i1, none, none -> none
      fabric.yield %0 : none
    }

    fabric.function_unit @fu_cond_br_i32(%cond: i1, %value: i32)
        -> (i32, i32) [latency = 1, interval = 1] {
      %0, %1 = handshake.cond_br %cond, %value : i32
      fabric.yield %0, %1 : i32, i32
    }

    fabric.function_unit @fu_cond_br_none(%cond: i1, %value: none)
        -> (none, none) [latency = 1, interval = 1] {
      %0, %1 = handshake.cond_br %cond, %value : none
      fabric.yield %0, %1 : none, none
    }

    fabric.function_unit @fu_cond_br_none_1(%cond: i1, %value: none)
        -> (none, none) [latency = 1, interval = 1] {
      %0, %1 = handshake.cond_br %cond, %value : none
      fabric.yield %0, %1 : none, none
    }

    fabric.function_unit @fu_load(%addr: index, %data: i32, %ctrl: none)
        -> (i32, index) [latency = 1, interval = 1] {
      %0, %1 = handshake.load [%addr] %data, %ctrl : index, i32
      fabric.yield %0, %1 : i32, index
    }

    fabric.function_unit @fu_load_1(%addr: index, %data: i32, %ctrl: none)
        -> (i32, index) [latency = 1, interval = 1] {
      %0, %1 = handshake.load [%addr] %data, %ctrl : index, i32
      fabric.yield %0, %1 : i32, index
    }

    fabric.function_unit @fu_addi_i32(%a: i32, %b: i32) -> (i32)
        [latency = 1, interval = 1] {
      %0 = arith.addi %a, %b : i32
      fabric.yield %0 : i32
    }

    fabric.function_unit @fu_addi_i64(%a: i64, %b: i64) -> (i64)
        [latency = 1, interval = 1] {
      %0 = arith.addi %a, %b : i64
      fabric.yield %0 : i64
    }

    fabric.function_unit @fu_addi_i32_1(%a: i32, %b: i32) -> (i32)
        [latency = 1, interval = 1] {
      %0 = arith.addi %a, %b : i32
      fabric.yield %0 : i32
    }

    fabric.function_unit @fu_addi_i32_2(%a: i32, %b: i32) -> (i32)
        [latency = 1, interval = 1] {
      %0 = arith.addi %a, %b : i32
      fabric.yield %0 : i32
    }

    fabric.function_unit @fu_cmpi(%a: i32, %b: i32) -> (i1)
        [latency = 1, interval = 1] {
      %0 = arith.cmpi sgt, %a, %b : i32
      fabric.yield %0 : i1
    }

    fabric.function_unit @fu_select_index(%cond: i1, %a: index, %b: index)
        -> (index) [latency = 1, interval = 1] {
      %0 = arith.select %cond, %a, %b : index
      fabric.yield %0 : index
    }

    fabric.function_unit @fu_mux_i32(%sel: index, %a: i32, %b: i32)
        -> (i32) [latency = 1, interval = 1] {
      %0 = handshake.mux %sel [%a, %b] : index, i32
      fabric.yield %0 : i32
    }

    fabric.function_unit @fu_mux_none(%sel: index, %a: none, %b: none)
        -> (none) [latency = 1, interval = 1] {
      %0 = handshake.mux %sel [%a, %b] : index, none
      fabric.yield %0 : none
    }

    fabric.function_unit @fu_index_cast_i32(%arg0: i32) -> (index)
        [latency = 1, interval = 1] {
      %0 = arith.index_cast %arg0 : i32 to index
      fabric.yield %0 : index
    }

    fabric.function_unit @fu_index_cast_i64(%arg0: i64) -> (index)
        [latency = 1, interval = 1] {
      %0 = arith.index_cast %arg0 : i64 to index
      fabric.yield %0 : index
    }

    fabric.yield
  }

  fabric.module @pairsum_tagged_temporal_domain(
      %mem0: memref<?xi32>,
      %n: !fabric.bits<64>,
      %ctrl: !fabric.bits<64>)
      -> (!fabric.bits<64>, !fabric.bits<64>) {
    %tag_n = fabric.add_tag %n {tag = 0 : i64}
        : !fabric.bits<64> -> !fabric.tagged<!fabric.bits<64>, i1>
    %tag_ctrl = fabric.add_tag %ctrl {tag = 0 : i64}
        : !fabric.bits<64> -> !fabric.tagged<!fabric.bits<64>, i1>

    %tpe0:4 = fabric.instance @tpe_pairsum(
        %tag_n, %tag_ctrl, %tsw_ld_data#0, %tsw_ld_done#0, %tsw_ld_data#1, %tsw_ld_done#1)
        {sym_name = "tpe_0"}
        : (!fabric.tagged<!fabric.bits<64>, i1>,
           !fabric.tagged<!fabric.bits<64>, i1>,
           !fabric.tagged<!fabric.bits<64>, i1>,
           !fabric.tagged<!fabric.bits<64>, i1>,
           !fabric.tagged<!fabric.bits<64>, i1>,
           !fabric.tagged<!fabric.bits<64>, i1>)
          -> (!fabric.tagged<!fabric.bits<64>, i1>,
              !fabric.tagged<!fabric.bits<64>, i1>,
              !fabric.tagged<!fabric.bits<64>, i1>,
              !fabric.tagged<!fabric.bits<64>, i1>)

    %sw_addr:1 = fabric.instance @addr_mux(%tpe0#0, %tpe0#1)
        {sym_name = "sw_0"}
        : (!fabric.tagged<!fabric.bits<64>, i1>,
           !fabric.tagged<!fabric.bits<64>, i1>)
          -> (!fabric.tagged<!fabric.bits<57>, i1>)

    %ext0:2 = fabric.extmemory @extmem_0
        [ldCount = 2, stCount = 0, lsqDepth = 0,
         memrefType = memref<?xi32>, numRegion = 1]
        (%mem0, %sw_addr#0)
        : (memref<?xi32>, !fabric.tagged<!fabric.bits<57>, i1>)
          -> (!fabric.tagged<!fabric.bits<64>, i1>,
              !fabric.tagged<!fabric.bits<64>, i1>)

    %tsw_ld_data:2 = fabric.instance @ld_data_demux(%ext0#0)
        {sym_name = "tsw_0"}
        : (!fabric.tagged<!fabric.bits<64>, i1>)
          -> (!fabric.tagged<!fabric.bits<64>, i1>,
              !fabric.tagged<!fabric.bits<64>, i1>)
    %tsw_ld_done:2 = fabric.instance @ld_done_demux(%ext0#1)
        {sym_name = "tsw_1"}
        : (!fabric.tagged<!fabric.bits<64>, i1>)
          -> (!fabric.tagged<!fabric.bits<64>, i1>,
              !fabric.tagged<!fabric.bits<64>, i1>)

    %sum_bits = fabric.del_tag %tpe0#2
        : !fabric.tagged<!fabric.bits<64>, i1> -> !fabric.bits<64>
    %done_bits = fabric.del_tag %tpe0#3
        : !fabric.tagged<!fabric.bits<64>, i1> -> !fabric.bits<64>

    fabric.yield %sum_bits, %done_bits : !fabric.bits<64>, !fabric.bits<64>
  }
}
)FCCADG";

} // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "pairsum_tagged_temporal ADG generator\n");

  auto parentPath = llvm::sys::path::parent_path(outputFile);
  if (!parentPath.empty())
    if (auto ec = llvm::sys::fs::create_directories(parentPath)) {
      llvm::errs() << "error: cannot create output directory: " << parentPath
                   << "\n";
      return 1;
    }

  std::error_code ec;
  llvm::raw_fd_ostream os(outputFile, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "error: cannot open output file: " << outputFile << "\n";
    return 1;
  }

  llvm::outs() << "Generating pairsum_tagged_temporal ADG -> " << outputFile
               << "\n";
  os << kPairsumTaggedTemporalADG;
  llvm::outs() << "Done.\n";
  return 0;
}
