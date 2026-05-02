// RUN: loom-cost-test %s --config %p/pure_determinism_a.yaml > %t.a
// RUN: loom-cost-test %s --config %p/pure_determinism_b.yaml > %t.b
// RUN: diff %t.a %t.b
// RUN: loom-cost-test %s --config %p/pure_determinism_a.yaml > %t.c
// RUN: diff %t.a %t.c
// RUN: cat %t.a | FileCheck %s

// CostModel must be pure: same FU + same cost weights produce
// byte-identical output, regardless of unrelated SynthConfig knobs
// (strategy / workers / restarts / seed) and across repeated runs.
//
// Configs A and B share identical mux/demux/carry penalties but differ
// in every other field. The two `diff` lines above are the actual
// determinism assertion; the FileCheck below pins the analytic values
// so a future weight-table refactor that breaks the formula still
// fails this test.
//
//   FU body: arith.addi i32 (baseUnit 1.0)         -> 1.0
//          + 2-port fabric.mux i32 (1.5 * 2 * 32)  -> 96.0
//          + 2-output fabric.demux i32 (1.5 * 2 * 32) -> 96.0
//          + dataflow.carry i32 (2.0 * 32)         -> 64.0
//          total = 1 + 96 + 96 + 64               = 257.0

fabric.module @det_full(%cond: !fabric.bits<32>,
                        %a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  fabric.pe [spatial] (%pcond = %cond : !fabric.bits<32>,
                       %pa = %a : !fabric.bits<32>,
                       %pb = %b : !fabric.bits<32>)
                      -> (!fabric.bits<32>, !fabric.bits<32>) {
    fabric.fu(%cc = %pcond : !fabric.bits<32> to !fabric.bits<1>,
              %x = %pa : !fabric.bits<32>,
              %y = %pb : !fabric.bits<32>)
              -> (!fabric.bits<32>, !fabric.bits<32>) {
      %m = fabric.mux %x, %y : !fabric.bits<32>
      %k = fabric.op [@arith.addi] (%m, %y)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %c = fabric.op [@dataflow.carry] (%cc, %k, %k)
           : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %d0, %d1 = fabric.demux %c : !fabric.bits<32> -> 2
      fabric.yield %d0, %d1 : !fabric.bits<32>, !fabric.bits<32>
    }
  }
  fabric.yield
}

// CHECK: cost det_full=2.570000e+02
