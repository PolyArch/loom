// RUN: loom-raise-opt --loom-lower-scf-to-dfg %s | FileCheck %s

// A pointer-carried residual loop with a then-only conditional store is the
// tail shape emitted by the CMSIS-NN q7 relu cleanup graph. The store is
// equivalent to selecting between the replacement value and the loaded value,
// so the loop can be streamed without preserving residual SCF in the graph.

// CHECK-LABEL: dataflow.graph.func private @g_conditional_store_loop
// CHECK: %[[IDX:.*]], %[[RWC:.*]] = dataflow.stream %arg1, %arg2, %arg3
// CHECK-SAME: cont_cond = ">"
// CHECK: %[[ZERO:.*]] = dataflow.invariant %[[RWC]], %arg4 : i8
// CHECK: %[[PTR_CARRY:.*]] = dataflow.carry %[[RWC]], %arg5,
// CHECK: %{{.*}}, %[[PTR_BODY:.*]] = dataflow.gate %[[RWC]], %[[PTR_CARRY]] : !llvm.ptr
// CHECK: %[[DATA:.*]], %{{.*}} = dataflow.load
// CHECK: %[[STORE_VALUE:.*]] = arith.{{(maxsi|select)}}
// CHECK: dataflow.store {{.*}} %[[STORE_VALUE]]
// CHECK-NOT: scf.for
// CHECK-NOT: scf.if
// CHECK: dataflow.graph.return
dataflow.graph.func private @g_conditional_store_loop(
    %ctrl: none, %lb: i16, %ub: i16, %step: i16, %zero: i8,
    %buf: !llvm.ptr) -> none {
  %r = scf.for %i = %lb to %ub step %step iter_args(%ptr = %buf)
      -> (!llvm.ptr) : i16 {
    %mem = builtin.unrealized_conversion_cast %ptr : !llvm.ptr to memref<?xi8>
    %c0 = dataflow.constant %ctrl {const_value = 0 : index} : index
    %data, %done = dataflow.load %mem[%c0] %ctrl : memref<?xi8>
    %neg = arith.cmpi slt, %data, %zero : i8
    scf.if %neg {
      %store_mem = builtin.unrealized_conversion_cast %ptr : !llvm.ptr
          to memref<?xi8>
      %store_idx = dataflow.constant %ctrl {const_value = 0 : index} : index
      %store = dataflow.store %store_mem[%store_idx] %zero %ctrl
          : memref<?xi8>
    }
    %next = llvm.getelementptr inbounds|nuw %ptr[1]
        : (!llvm.ptr) -> !llvm.ptr, i8
    scf.yield %next : !llvm.ptr
  } {loom.stream_cont_cond = ">"}
  dataflow.graph.return %ctrl : none
}
