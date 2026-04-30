// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// FU with 4 input ports %w, %x, %y, %z. A fabric.mux selects one of (%w,
// %x) and feeds it together with %y into an arith.addi; %z never reaches
// any live compute. The enumerator must therefore emit per-config
// subgraph templates that expose only the live FU input ports rather
// than carrying along the dead ones.
//
// For mux.sel=0 the live FU inputs are %w and %y; for mux.sel=1 they are
// %x and %y. Either way the materialized subgraph signature is
// (i32, i32) -> i32 with exactly two block arguments. The remaining
// fabric.mux modes (discard / disconnect) leave the addi without a left
// operand and produce no candidate.

// CHECK-LABEL: fabric.module @fu_internal_mux_2of4
fabric.module @fu_internal_mux_2of4(%a : !fabric.bits<32>, %b : !fabric.bits<32>, %c : !fabric.bits<32>, %d : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>,
                    %pc = %c : !fabric.bits<32>,
                    %pd = %d : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%w = %pa : !fabric.bits<32>,
              %x = %pb : !fabric.bits<32>,
              %y = %pc : !fabric.bits<32>,
              %z = %pd : !fabric.bits<32>) -> !fabric.bits<32> {
      %m = fabric.mux %w, %x : !fabric.bits<32>
      %r0 = fabric.op [@arith.addi] (%m, %y)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %r0 : !fabric.bits<32>
    }
  }
  fabric.yield
}

// sel=0 and sel=1 produce graph-isomorphic 2-input subgraphs (an addi
// of two distinct block args). Dedup keeps only the lex-smallest, so
// exactly one wrapper / one config is emitted.
// CHECK: func.func private @fu0_subgraph_0(%{{[^,]*}}: i32, %{{[^,]*}}: i32) -> i32
// CHECK: dataflow.subgraph(%{{.*}} = %{{.*}} : i32, %{{.*}} = %{{.*}} : i32) -> i32
// CHECK-SAME: mux#0{sel=0,discard=false,disconnect=false}
// CHECK:   arith.addi
// CHECK:   dataflow.yield

// No second wrapper is emitted (the sel=1 effective config is
// isomorphic to sel=0 and therefore deduped).
// CHECK-NOT: func.func private @fu0_subgraph_1
// CHECK-NOT: mux#0{sel=1,discard=false,disconnect=false}

// No 4-input subgraph signature must ever be emitted for this FU.
// CHECK-NOT: func.func private @fu0_subgraph_{{[0-9]+}}(%{{[^,]*}}: i32, %{{[^,]*}}: i32, %{{[^,]*}}: i32, %{{[^,]*}}: i32)

// The discard / disconnect mux configurations starve the addi and must
// not produce a candidate.
// CHECK-NOT: mux#0{sel=0,discard=true,disconnect=false}
// CHECK-NOT: mux#0{sel=1,discard=true,disconnect=false}
// CHECK-NOT: mux#0{sel=0,discard=false,disconnect=true}
