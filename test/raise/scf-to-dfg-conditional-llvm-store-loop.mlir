// RUN: loom-raise-opt --loom-lower-scf-to-dfg %s | FileCheck %s

// CMSIS-NN q7 relu tails arrive at reduction lowering as an LLVM
// load plus then-only LLVM store. The reduction pass must still stream
// the loop so the later memory/control passes can tokenize the memory
// effects and rewrite the conditional store into a select-fed
// dataflow.store.

// CHECK-LABEL: dataflow.graph.func private @g_conditional_llvm_store_loop
// CHECK: %[[IDX:.*]], %[[RWC:.*]] = dataflow.stream %arg1, %arg2, %arg3
// CHECK-SAME: cont_cond = ">"
// CHECK: %[[ZERO:.*]] = dataflow.invariant %[[RWC]], %arg4 : i8
// CHECK: %[[PTR_CARRY:.*]] = dataflow.carry %[[RWC]], %arg5,
// CHECK: %{{.*}}, %[[PTR_BODY:.*]] = dataflow.gate %[[RWC]], %[[PTR_CARRY]] : !llvm.ptr
// CHECK: %[[DATA:.*]], %{{.*}} = dataflow.load
// CHECK: %[[STORE_MEM:.*]] = builtin.unrealized_conversion_cast %[[PTR_BODY]] : !llvm.ptr to memref<?xi8>
// CHECK: %[[STORE_VALUE:.*]] = arith.{{(maxsi|select)}}
// CHECK: dataflow.store %[[STORE_MEM]]{{.*}} %[[STORE_VALUE]]
// CHECK-NOT: scf.for
// CHECK-NOT: scf.if
// CHECK-NOT: llvm.load
// CHECK-NOT: llvm.store
// CHECK: dataflow.graph.return
dataflow.graph.func private @g_conditional_llvm_store_loop(
    %ctrl: none, %lb: i16, %ub: i16, %step: i16, %zero: i8,
    %buf: !llvm.ptr) -> none {
  %r = scf.for %i = %lb to %ub step %step iter_args(%ptr = %buf)
      -> (!llvm.ptr) : i16 {
    %data = llvm.load %ptr : !llvm.ptr -> i8
    %neg = arith.cmpi slt, %data, %zero : i8
    scf.if %neg {
      llvm.store %zero, %ptr : i8, !llvm.ptr
    }
    %next = llvm.getelementptr inbounds|nuw %ptr[1]
        : (!llvm.ptr) -> !llvm.ptr, i8
    scf.yield %next : !llvm.ptr
  } {loom.stream_cont_cond = ">"}
  dataflow.graph.return %ctrl : none
}
