// RUN: loom %s | loom | FileCheck %s

// -----------------------------------------------------------------------------
// Single-op fabric.op: arith.muli (singleton group), pure hardware (no sw_configs).
// -----------------------------------------------------------------------------

// CHECK-LABEL: @op_single_muli_hw
func.func @op_single_muli_hw(%a: !fabric.bits<32>, %b: !fabric.bits<32>) -> !fabric.bits<32> {
  // CHECK: fabric.op [@arith.muli](%{{.*}}, %{{.*}}) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %0 = fabric.op [@arith.muli] (%a, %b) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  return %0 : !fabric.bits<32>
}

// -----------------------------------------------------------------------------
// Multi-op group {arith.addi, arith.subi}, programmed to subi.
// -----------------------------------------------------------------------------

// CHECK-LABEL: @op_addi_subi_programmed
func.func @op_addi_subi_programmed(%a: !fabric.bits<32>, %b: !fabric.bits<32>) -> !fabric.bits<32> {
  // CHECK: fabric.op [@arith.addi, @arith.subi](%{{.*}}, %{{.*}}) {sw_configs = {op_sel = "arith.subi"}}
  %0 = fabric.op [@arith.addi, @arith.subi] (%a, %b)
       {sw_configs = {op_sel = "arith.subi"}}
       : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  return %0 : !fabric.bits<32>
}

// -----------------------------------------------------------------------------
// Multi-op group, pure hardware (sw_configs absent => not programmed).
// -----------------------------------------------------------------------------

// CHECK-LABEL: @op_divrem_pure_hardware
func.func @op_divrem_pure_hardware(%a: !fabric.bits<64>, %b: !fabric.bits<64>) -> !fabric.bits<64> {
  // CHECK: fabric.op [@arith.divsi, @arith.remsi](%{{.*}}, %{{.*}}) : (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
  %0 = fabric.op [@arith.divsi, @arith.remsi] (%a, %b)
       : (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
  return %0 : !fabric.bits<64>
}

// -----------------------------------------------------------------------------
// arith.cmpi: 2 in same width, 1 out i1 (== bits<1>); predicate via hw/sw params.
// -----------------------------------------------------------------------------

// CHECK-LABEL: @op_cmpi
func.func @op_cmpi(%a: !fabric.bits<32>, %b: !fabric.bits<32>) -> !fabric.bits<1> {
  // CHECK: fabric.op [@arith.cmpi]
  %0 = fabric.op [@arith.cmpi] (%a, %b)
       {hw_params = [{predicate = ["eq", "ne", "slt", "sgt"]}],
        sw_configs = {predicate = "slt"}}
       : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
  return %0 : !fabric.bits<1>
}

// -----------------------------------------------------------------------------
// dataflow.stream programmed: 3 in T, out (T, i1).
// -----------------------------------------------------------------------------

// CHECK-LABEL: @op_stream_programmed
func.func @op_stream_programmed(%lb: !fabric.bits<32>, %ub: !fabric.bits<32>, %step: !fabric.bits<32>)
    -> (!fabric.bits<32>, !fabric.bits<1>) {
  // CHECK: fabric.op [@dataflow.stream]
  %i, %r = fabric.op [@dataflow.stream] (%lb, %ub, %step)
           {hw_params = [{step_op = ["+=", "/=", "*="], cont_cond = ["<", ">"]}],
            sw_configs = {step_op = "+=", cont_cond = "<"}}
           : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
             -> (!fabric.bits<32>, !fabric.bits<1>)
  return %i, %r : !fabric.bits<32>, !fabric.bits<1>
}

// -----------------------------------------------------------------------------
// dataflow.constant: 1 in bits<0> (none token), 1 out value, sw_configs only.
// -----------------------------------------------------------------------------

// CHECK-LABEL: @op_constant
func.func @op_constant(%ctrl: !fabric.bits<0>) -> !fabric.bits<32> {
  // CHECK: fabric.op [@dataflow.constant](%{{.*}}) {sw_configs = {const_hex_value = "0xdeadbeef"}}
  %0 = fabric.op [@dataflow.constant] (%ctrl)
       {sw_configs = {const_hex_value = "0xdeadbeef"}}
       : (!fabric.bits<0>) -> !fabric.bits<32>
  return %0 : !fabric.bits<32>
}

// -----------------------------------------------------------------------------
// dataflow.sync: variadic; verifier defers strict count to bitmask interpretation.
// -----------------------------------------------------------------------------

// CHECK-LABEL: @op_sync
func.func @op_sync(%a: !fabric.bits<32>, %b: !fabric.bits<32>, %c: !fabric.bits<32>, %d: !fabric.bits<32>)
    -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) {
  // CHECK: fabric.op [@dataflow.sync]
  // CHECK-SAME: sw_configs = {bitmask = "1101"}
  %x, %y, %z = fabric.op [@dataflow.sync] (%a, %b, %c, %d)
               {sw_configs = {bitmask = "1101"}}
               : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
                 -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  return %x, %y, %z : !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>
}

// -----------------------------------------------------------------------------
// dataflow.gate: 2 in (i1, T), 2 out (i1, T).
// -----------------------------------------------------------------------------

// CHECK-LABEL: @op_gate
func.func @op_gate(%bc: !fabric.bits<1>, %bv: !fabric.bits<32>) -> (!fabric.bits<1>, !fabric.bits<32>) {
  // CHECK: fabric.op [@dataflow.gate]
  %ac, %av = fabric.op [@dataflow.gate] (%bc, %bv)
             : (!fabric.bits<1>, !fabric.bits<32>) -> (!fabric.bits<1>, !fabric.bits<32>)
  return %ac, %av : !fabric.bits<1>, !fabric.bits<32>
}
