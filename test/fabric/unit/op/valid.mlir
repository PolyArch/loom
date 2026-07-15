// RUN: loom %s | loom | FileCheck %s

// Note: fabric.op must live inside fabric.fu, which lives inside
// fabric.pe, which lives inside fabric.module. PE/FU external ports
// must be uniform !fabric.bits<W>; internal fabric.op ports may be any
// fabric.bits<N>. Tests whose inner op has heterogeneous port widths are
// wrapped with the FU exposing only the input-side width to the PE and a
// `-> ()` output (the inner op's mixed-width result is consumed only inside
// the FU) so that the test continues to exercise the inner op verifier.

// -----------------------------------------------------------------------------
// Single-op fabric.op: arith.muli (singleton group), pure hardware (no sw_configs).
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @op_single_muli_hw
fabric.module @op_single_muli_hw(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      // CHECK: fabric.op [@arith.muli](%{{.*}}, %{{.*}}) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %0 = fabric.op [@arith.muli] (%fa, %fb) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %0 : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----------------------------------------------------------------------------
// Multi-op group {arith.addi, arith.subi}, programmed to subi.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @op_addi_subi_programmed
fabric.module @op_addi_subi_programmed(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      // CHECK: fabric.op [@arith.addi, @arith.subi](%{{.*}}, %{{.*}}) {sw_configs = {op_sel = "arith.subi"}}
      %0 = fabric.op [@arith.addi, @arith.subi] (%fa, %fb)
           {sw_configs = {op_sel = "arith.subi"}}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %0 : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----------------------------------------------------------------------------
// LLVM packed saturating 16-bit add/sub share one lane-wise datapath.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @op_llvm_packed_sat16
fabric.module @op_llvm_packed_sat16(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      // CHECK: fabric.op [@llvm.arm.qadd16, @llvm.arm.qsub16]
      %0 = fabric.op [@llvm.arm.qadd16, @llvm.arm.qsub16] (%fa, %fb)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %0 : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----------------------------------------------------------------------------
// LLVM integer casts share one bit extraction / fill datapath.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @op_llvm_int_casts
fabric.module @op_llvm_int_casts(%a : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
      // CHECK: fabric.op [@llvm.trunc, @llvm.sext, @llvm.zext]
      %0 = fabric.op [@llvm.trunc, @llvm.sext, @llvm.zext] (%fa)
           : (!fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %0 : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----------------------------------------------------------------------------
// Multi-op group, pure hardware (sw_configs absent => not programmed).
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @op_divrem_pure_hardware
fabric.module @op_divrem_pure_hardware(%a : !fabric.bits<64>, %b : !fabric.bits<64>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<64>,
                    %pb = %b : !fabric.bits<64>) -> !fabric.bits<64> {
    fabric.fu(%fa = %pa : !fabric.bits<64>,
              %fb = %pb : !fabric.bits<64>) -> !fabric.bits<64> {
      // CHECK: fabric.op [@arith.divsi, @arith.remsi](%{{.*}}, %{{.*}}) : (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
      %0 = fabric.op [@arith.divsi, @arith.remsi] (%fa, %fb)
           : (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
      fabric.yield %0 : !fabric.bits<64>
    }
  }
  fabric.yield
}

// -----------------------------------------------------------------------------
// arith.cmpi: 2 in same width, 1 out i1 (== bits<1>); predicate via hw/sw params.
// PE/FU at the input width (bits<32>); the bits<1> result stays inside the FU.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @op_cmpi
fabric.module @op_cmpi(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>) -> () {
      // CHECK: fabric.op [@arith.cmpi]
      %0 = fabric.op [@arith.cmpi] (%fa, %fb)
           {hw_params = [{predicate = ["eq", "ne", "slt", "sgt"]}],
            sw_configs = {predicate = "slt"}}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
      fabric.yield
    }
  }
  fabric.yield
}

// -----------------------------------------------------------------------------
// dataflow.stream programmed: 3 in T, out (T, i1).
// PE/FU at bits<32>; the bits<1> rwc result is consumed only inside the FU.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @op_stream_programmed
fabric.module @op_stream_programmed(%lb : !fabric.bits<32>, %ub : !fabric.bits<32>, %step : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %lb : !fabric.bits<32>,
                    %pb = %ub : !fabric.bits<32>,
                    %pc = %step : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>,
              %fc = %pc : !fabric.bits<32>) -> () {
      // CHECK: fabric.op [@dataflow.stream]
      %i, %r = fabric.op [@dataflow.stream] (%fa, %fb, %fc)
               {hw_params = [{step_op = ["+=", "/=", "*="], cont_cond = ["<", ">"]}],
                sw_configs = {step_op = "+=", cont_cond = "<"}}
               : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
                 -> (!fabric.bits<32>, !fabric.bits<1>)
      fabric.yield
    }
  }
  fabric.yield
}

// -----------------------------------------------------------------------------
// dataflow.constant: 1 in bits<0> (none token), 1 out value, sw_configs only.
// PE/FU at bits<0>; the bits<32> result is consumed only inside the FU.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @op_constant
fabric.module @op_constant(%ctrl : !fabric.bits<0>) {
  fabric.pe [spatial] (%pa = %ctrl : !fabric.bits<0>) -> !fabric.bits<0> {
    fabric.fu(%fa = %pa : !fabric.bits<0>) -> () {
      // CHECK: fabric.op [@dataflow.constant](%{{.*}}) {sw_configs = {const_hex_value = "0xdeadbeef"}}
      %0 = fabric.op [@dataflow.constant] (%fa)
           {sw_configs = {const_hex_value = "0xdeadbeef"}}
           : (!fabric.bits<0>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  fabric.yield
}

// -----------------------------------------------------------------------------
// dataflow.sync: variadic; verifier defers strict count to bitmask interpretation.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @op_sync
fabric.module @op_sync(%a : !fabric.bits<32>, %b : !fabric.bits<32>, %c : !fabric.bits<32>, %d : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>,
                    %pc = %c : !fabric.bits<32>,
                    %pd = %d : !fabric.bits<32>)
                   -> (!fabric.bits<32>, !fabric.bits<32>,
                       !fabric.bits<32>, !fabric.bits<32>) {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>,
              %fc = %pc : !fabric.bits<32>,
              %fd = %pd : !fabric.bits<32>)
              -> (!fabric.bits<32>, !fabric.bits<32>,
                  !fabric.bits<32>, !fabric.bits<32>) {
      // CHECK: fabric.op [@dataflow.sync]
      // CHECK-SAME: sw_configs = {bitmask = "1101"}
      %w, %x, %y, %z = fabric.op [@dataflow.sync] (%fa, %fb, %fc, %fd)
                       {sw_configs = {bitmask = "1101"}}
                       : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
                         -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
      fabric.yield %w, %x, %y, %z : !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----------------------------------------------------------------------------
// dataflow.mux: 1 sel + 2 data (sel is bits<1>) -> 1 out.
// PE/FU at bits<32>; sel is generated internally (bits<1>) and consumed
// inside the FU. The bits<32> result stays inside the FU as well.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @op_mux2
fabric.module @op_mux2(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>) -> () {
      %sel = fabric.op [@arith.cmpi] (%fa, %fb)
             : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
      // CHECK: fabric.op [@dataflow.mux]
      %0 = fabric.op [@dataflow.mux] (%sel, %fa, %fb)
           : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  fabric.yield
}

// -----------------------------------------------------------------------------
// dataflow.mux: 1 sel + 4 data (sel is bits<32> = index width default).
// PE/FU at bits<16> (data width); sel is generated internally as bits<32>.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @op_mux4
fabric.module @op_mux4(%a : !fabric.bits<16>, %b : !fabric.bits<16>, %c : !fabric.bits<16>, %d : !fabric.bits<16>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<16>,
                    %pb = %b : !fabric.bits<16>,
                    %pc = %c : !fabric.bits<16>,
                    %pd = %d : !fabric.bits<16>) -> !fabric.bits<16> {
    fabric.fu(%fa = %pa : !fabric.bits<16>,
              %fb = %pb : !fabric.bits<16>,
              %fc = %pc : !fabric.bits<16>,
              %fd = %pd : !fabric.bits<16>) -> () {
      %sel = fabric.op [@arith.sitofp] (%fa)
             : (!fabric.bits<16>) -> !fabric.bits<32>
      // CHECK: fabric.op [@dataflow.mux]
      %0 = fabric.op [@dataflow.mux] (%sel, %fa, %fb, %fc, %fd)
           : (!fabric.bits<32>, !fabric.bits<16>, !fabric.bits<16>,
              !fabric.bits<16>, !fabric.bits<16>) -> !fabric.bits<16>
      fabric.yield
    }
  }
  fabric.yield
}

// -----------------------------------------------------------------------------
// dataflow.demux: 1 sel + 1 data -> 3 outs (sel is bits<32> = index).
// PE/FU at bits<8> (data width); sel is generated internally as bits<32>.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @op_demux3
fabric.module @op_demux3(%in : !fabric.bits<8>) {
  fabric.pe [spatial] (%pa = %in : !fabric.bits<8>) -> !fabric.bits<8> {
    fabric.fu(%fa = %pa : !fabric.bits<8>) -> () {
      %sel = fabric.op [@arith.sitofp] (%fa)
             : (!fabric.bits<8>) -> !fabric.bits<32>
      // CHECK: fabric.op [@dataflow.demux]
      %a, %b, %c = fabric.op [@dataflow.demux] (%sel, %fa)
                   : (!fabric.bits<32>, !fabric.bits<8>)
                     -> (!fabric.bits<8>, !fabric.bits<8>, !fabric.bits<8>)
      fabric.yield
    }
  }
  fabric.yield
}

// -----------------------------------------------------------------------------
// dataflow.gate: 2 in (i1, T), 2 out (i1, T).
// PE/FU at bits<32>; bc (bits<1>) is generated internally.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @op_gate
fabric.module @op_gate(%bv : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %bv : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> () {
      %bc = fabric.op [@arith.cmpi] (%fa, %fa)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
      // CHECK: fabric.op [@dataflow.gate]
      %ac, %av = fabric.op [@dataflow.gate] (%bc, %fa)
                 : (!fabric.bits<1>, !fabric.bits<32>) -> (!fabric.bits<1>, !fabric.bits<32>)
      fabric.yield
    }
  }
  fabric.yield
}

// CHECK-LABEL: fabric.module @op_normalized_hw_modes
fabric.module @op_normalized_hw_modes(%a : !fabric.bits<32>,
                                      %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                      %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>) -> () {
      // CHECK: sw_configs = {mode = 1 : i32}
      %v = fabric.op [@arith.addi, @arith.subi] (%fa, %fb)
           {hw_params = [
             {op = @arith.addi, function_type = (i32, i32) -> i32,
              input_ports = [0 : i32, 1 : i32],
              output_ports = [0 : i32], attributes = {}},
             {op = @arith.subi, function_type = (i32, i32) -> i32,
              input_ports = [0 : i32, 1 : i32],
              output_ports = [0 : i32], attributes = {}}
           ], sw_configs = {mode = 1 : i32}}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  fabric.yield
}
