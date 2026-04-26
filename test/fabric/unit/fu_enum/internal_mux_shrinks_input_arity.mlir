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

// CHECK-LABEL: @fu_internal_mux_2of4
func.func @fu_internal_mux_2of4(%a: !fabric.bits<32>, %b: !fabric.bits<32>,
                                 %c: !fabric.bits<32>, %d: !fabric.bits<32>) {
  %r = fabric.fu(%w = %a : !fabric.bits<32>,
                 %x = %b : !fabric.bits<32>,
                 %y = %c : !fabric.bits<32>,
                 %z = %d : !fabric.bits<32>) -> !fabric.bits<32> {
    %m = fabric.mux %w, %x : !fabric.bits<32>
    %r0 = fabric.op [@arith.addi] (%m, %y)
          : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %r0 : !fabric.bits<32>
  }

  // sel=0 candidate: live inputs are FU args 0 and 2, lifted to a
  // 2-input/1-output subgraph. The wrapping func.func captures the
  // canonical signature.
  // CHECK: func.func private @fu0_subgraph_0(%{{[^,]*}}: i32, %{{[^,]*}}: i32) -> i32
  // CHECK: dataflow.subgraph(%{{.*}} = %{{.*}} : i32, %{{.*}} = %{{.*}} : i32) -> i32
  // CHECK-SAME: mux#0{sel=0,discard=false,disconnect=false}
  // CHECK:   arith.addi
  // CHECK:   dataflow.yield

  // sel=1 candidate: live inputs are FU args 1 and 2, still 2-input.
  // CHECK: func.func private @fu0_subgraph_1(%{{[^,]*}}: i32, %{{[^,]*}}: i32) -> i32
  // CHECK: dataflow.subgraph(%{{.*}} = %{{.*}} : i32, %{{.*}} = %{{.*}} : i32) -> i32
  // CHECK-SAME: mux#0{sel=1,discard=false,disconnect=false}
  // CHECK:   arith.addi
  // CHECK:   dataflow.yield

  // No 4-input subgraph signature must ever be emitted for this FU.
  // CHECK-NOT: func.func private @fu0_subgraph_{{[0-9]+}}(%{{[^,]*}}: i32, %{{[^,]*}}: i32, %{{[^,]*}}: i32, %{{[^,]*}}: i32)

  // The discard / disconnect mux configurations starve the addi and must
  // not produce a candidate.
  // CHECK-NOT: mux#0{sel=0,discard=true,disconnect=false}
  // CHECK-NOT: mux#0{sel=1,discard=true,disconnect=false}
  // CHECK-NOT: mux#0{sel=0,discard=false,disconnect=true}

  return
}
