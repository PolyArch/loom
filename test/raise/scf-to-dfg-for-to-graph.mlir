// RUN: loom-raise-opt --loom-lower-for-to-graph %s | FileCheck %s
// RUN: loom-raise-opt --loom-lower-for-to-graph %s | FileCheck %s --check-prefix=NO-CARRIED

// scf.for with iter_args inside a dataflow.thread body lowers to a
// sibling dataflow.graph.func definition + a dataflow.graph.launch
// at the cut site. The host_reduction case below also exercises the
// host-scope wrap path: a stand-alone reduction at host scope is
// wrapped in a synthetic 1x1 thread before being promoted.

// The thread carries the spec-mandated thread_ctrl slot, and the
// graph.launch consumes it directly as ctrl_in (no ub.poison).
// CHECK-LABEL: dataflow.thread private @t_existing
// CHECK-SAME: ctrl (%[[CTRL:.*]]: none)
// CHECK: dataflow.graph.launch @g_t_existing_0(%[[CTRL]]
// CHECK-NOT: ub.poison : none
// CHECK-NOT: scf.for {{.*}} iter_args
dataflow.thread private @t_existing(%buf: memref<?xf32>, %n: index) ctrl (%c: none) {
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %f0) -> (f32) {
    %v = memref.load %buf[%i] : memref<?xf32>
    %s = arith.addf %acc, %v : f32
    scf.yield %s : f32
  }
  dataflow.thread.yield
}

// Unused loop results must not become graph user-data returns. Pointer-walking
// loops often carry source/destination pointers only to drive memory accesses;
// if the enclosing thread does not use the final pointers, exposing them as
// graph results forces downstream mapping to model fake live pointer outputs.
// CHECK-LABEL: dataflow.thread private @t_unused_ptr_walk
// CHECK: dataflow.graph.launch @g_t_unused_ptr_walk_0(%{{.*}}) : (none, index, index, index, !llvm.ptr, !llvm.ptr) -> none
// CHECK-LABEL: dataflow.thread private @t_forall_store
// CHECK-SAME: ctrl (%[[FORALL_CTRL:.*]]: none) iv
// CHECK: dataflow.graph.launch @g_t_forall_store_0(%[[FORALL_CTRL]]
// CHECK-NOT: scf.forall
// CHECK-LABEL: func.func @host_reduction
// CHECK: dataflow.thread.launch @t_host_reduction_red_0
dataflow.thread private @t_unused_ptr_walk(%src: !llvm.ptr, %dst: !llvm.ptr,
                                           %n: index) ctrl (%c: none) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %unused:2 = scf.for %i = %c0 to %n step %c1 iter_args(%s = %src, %d = %dst)
      -> (!llvm.ptr, !llvm.ptr) {
    %s_next = llvm.getelementptr %s[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %d_next = llvm.getelementptr %d[4] : (!llvm.ptr) -> !llvm.ptr, i8
    scf.yield %s_next, %d_next : !llvm.ptr, !llvm.ptr
  }
  dataflow.thread.yield
}

// Effect-form scf.forall inside a thread is still a SpatialCore graph body.
// It must be extracted as a structured graph.func rather than leaving the
// kernel body stranded in the thread.
dataflow.thread private @t_forall_store(%src: memref<?xi32>, %dst: memref<?xi32>,
                                        %n: index) ctrl (%c: none) iv (%tile: index) {
  %c4 = arith.constant 4 : index
  %base = arith.muli %tile, %c4 : index
  scf.forall (%lane) in (4) {
    %idx = arith.addi %base, %lane : index
    %v = memref.load %src[%idx] : memref<?xi32>
    memref.store %v, %dst[%idx] : memref<?xi32>
  }
  dataflow.thread.yield
}

func.func @host_reduction(%buf: memref<?xf32>, %n: index) -> f32 {
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %f0) -> (f32) {
    %v = memref.load %buf[%i] : memref<?xf32>
    %s = arith.addf %acc, %v : f32
    scf.yield %s : f32
  }
  return %r : f32
}

// Structured lane-local control in a thread is still a valid graph body
// when it contains no nested launch boundary. The graph keeps the
// scf.if/scf.while structure so simulator and mapper handle the actual
// control flow instead of flattening it in the front end.
// CHECK-LABEL: dataflow.thread private @t_structured_while
// CHECK: dataflow.graph.launch @g_t_structured_while_0
dataflow.thread private @t_structured_while(%src: memref<?xi32>, %dst: memref<?xi32>,
                                            %n: index) ctrl (%c: none) iv (%lane: index) {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %value = memref.load %src[%lane] : memref<?xi32>
  %is_zero = arith.cmpi eq, %value, %zero : i32
  %count = scf.if %is_zero -> (i32) {
    scf.yield %zero : i32
  } else {
    %result:2 = scf.while (%v = %value, %acc = %zero) : (i32, i32) -> (i32, i32) {
      %more = arith.cmpi ne, %v, %zero : i32
      scf.condition(%more) %v, %acc : i32, i32
    } do {
    ^bb0(%v_next: i32, %acc_next: i32):
      %bit = arith.andi %v_next, %one : i32
      %updated = arith.addi %acc_next, %bit : i32
      %shifted = arith.shrui %v_next, %one : i32
      scf.yield %shifted, %updated : i32, i32
    }
    scf.yield %result#1 : i32
  }
  memref.store %count, %dst[%lane] : memref<?xi32>
  dataflow.thread.yield
}

// Structured lane-local switch control is also part of the graph body. Search
// kernels use this shape to choose between the next scan state and the selected
// interval for the current lane.
// CHECK-LABEL: dataflow.thread private @t_structured_index_switch
// CHECK: dataflow.graph.launch @g_t_structured_index_switch_0
dataflow.thread private @t_structured_index_switch(%src: memref<?xi32>, %dst: memref<?xi32>,
                                                   %n: index) ctrl (%c: none) iv (%lane: index) {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %tag = memref.load %src[%lane] : memref<?xi32>
  %selector = arith.index_castui %tag : i32 to index
  %selected = scf.index_switch %selector -> i32
  case 0 {
    scf.yield %zero : i32
  }
  default {
    scf.yield %one : i32
  }
  memref.store %selected, %dst[%lane] : memref<?xi32>
  dataflow.thread.yield
}

// A host-scope memcpy-only function is still an accelerator candidate:
// the compiler must expose a graph.func surface so graph-memory lowering can
// turn the copy into real stream load/store ops. This is intentionally a
// graph-only extraction; there is no synthetic host graph.launch.
// CHECK-LABEL: func.func @standalone_memcpy
// CHECK: llvm.intr.memcpy
func.func @standalone_memcpy(%src: !llvm.ptr, %dst: !llvm.ptr, %n: i32) {
  "llvm.intr.memcpy"(%dst, %src, %n)
    <{arg_attrs = [{llvm.align = 1 : i64}, {llvm.align = 1 : i64}, {}],
       isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
  return
}

// Straight-line address arithmetic feeding a single memcpy is also a
// graph-only accelerator candidate. The extracted graph keeps the scalar
// address math so graph-memory lowering can turn the copy into real stream
// load/store ops with an offset destination.
// CHECK-LABEL: func.func @standalone_offset_memcpy
// CHECK: llvm.intr.memcpy
func.func @standalone_offset_memcpy(%src: !llvm.ptr, %channels: i16,
                                    %height: i16, %width: i16,
                                    %dst: !llvm.ptr, %offset: i32) {
  %0 = llvm.zext %channels : i16 to i32
  %1 = llvm.zext %height : i16 to i32
  %2 = arith.muli %1, %0 : i32
  %3 = llvm.zext %width : i16 to i32
  %copy_bytes = arith.muli %2, %3 : i32
  %dst_offset = arith.muli %offset, %2 : i32
  %dst_at = llvm.getelementptr inbounds|nuw %dst[%dst_offset]
      : (!llvm.ptr, i32) -> !llvm.ptr, i8
  "llvm.intr.memcpy"(%dst_at, %src, %copy_bytes)
    <{arg_attrs = [{llvm.align = 1 : i64}, {llvm.align = 1 : i64}, {}],
       isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
  return
}

// A standalone private structured kernel with no results is also an
// accelerator candidate. The compiler exposes a graph-only surface while
// leaving the original callable function intact for host semantics.
// CHECK-LABEL: func.func private @scatter_add_candidate
// CHECK: scf.for
// CHECK: scf.if
func.func private @scatter_add_candidate(%src: !llvm.ptr, %idx: !llvm.ptr,
                                         %dst: !llvm.ptr) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c4 = arith.constant 4 : i64
  %limit = arith.constant 8 : i32
  scf.for %i = %c0 to %c4 step %c1 : i64 {
    %idx_ptr = llvm.getelementptr %idx[%i]
        : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
    %slot = llvm.load %idx_ptr : !llvm.ptr -> i32
    %ok = arith.cmpi ult, %slot, %limit : i32
    scf.if %ok {
      %src_ptr = llvm.getelementptr %src[%i]
          : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
      %value = llvm.load %src_ptr : !llvm.ptr -> i32
      %slot64 = llvm.zext %slot : i32 to i64
      %dst_ptr = llvm.getelementptr %dst[%slot64]
          : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
      %old = llvm.load %dst_ptr : !llvm.ptr -> i32
      %sum = arith.addi %old, %value : i32
      llvm.store %sum, %dst_ptr : i32, !llvm.ptr
    }
  }
  return
}

// Some kernels lower a no-op trip-count guard around the actual structured
// loop. The guard is still part of the SpatialCore graph boundary when its
// non-empty branch contains one top-level loop and no nested launch.
// CHECK-LABEL: func.func private @guarded_loop_candidate
// CHECK: scf.if
func.func private @guarded_loop_candidate(%src: !llvm.ptr, %dst: !llvm.ptr,
                                          %n: i32, %pass: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %distance = arith.shli %c1_i32, %pass : i32
  %empty = arith.cmpi eq, %n, %c0_i32 : i32
  scf.if %empty {
  } else {
    %limit = llvm.zext %n : i32 to i64
    scf.for %i = %c0_i64 to %limit step %c1_i64 : i64 {
      %idx = llvm.trunc %i : i64 to i32
      %partner = arith.addi %idx, %distance : i32
      %in_bounds = arith.cmpi ult, %partner, %n : i32
      scf.if %in_bounds {
        %src_ptr = llvm.getelementptr %src[%i]
            : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
        %value = llvm.load %src_ptr : !llvm.ptr -> f32
        %dst_idx = llvm.zext %partner : i32 to i64
        %dst_ptr = llvm.getelementptr %dst[%dst_idx]
            : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
        llvm.store %value, %dst_ptr : f32, !llvm.ptr
      }
    }
  }
  return
}

// Loop-carried state belongs to the thread/reduction path. A guarded
// iter_args loop must not also receive a standalone graph-only clone.
// NO-CARRIED: module
// NO-CARRIED-NOT: dataflow.graph.func private @g_guarded_carried_loop_candidate_0
func.func private @guarded_carried_loop_candidate(%src: !llvm.ptr,
                                                  %dst: !llvm.ptr,
                                                  %n: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %empty = arith.cmpi eq, %n, %c0_i32 : i32
  scf.if %empty {
  } else {
    %next:2 = scf.for %i = %n to %c0_i32 step %c1_i32
        iter_args(%s = %src, %d = %dst) -> (!llvm.ptr, !llvm.ptr) : i32 {
      %value = llvm.load %s : !llvm.ptr -> f32
      llvm.store %value, %d : f32, !llvm.ptr
      %s_next = llvm.getelementptr %s[4] : (!llvm.ptr) -> !llvm.ptr, i8
      %d_next = llvm.getelementptr %d[4] : (!llvm.ptr) -> !llvm.ptr, i8
      scf.yield %s_next, %d_next : !llvm.ptr, !llvm.ptr
    }
  }
  return
}

// A standalone structured kernel that computes a scalar through result-bearing
// control and writes it to an out pointer is an accelerator candidate.
// CHECK-LABEL: func.func private @structured_outparam_candidate
// CHECK: scf.if
// CHECK: scf.while
// CHECK: llvm.store
func.func private @structured_outparam_candidate(%src: !llvm.ptr,
                                                 %dst: !llvm.ptr,
                                                 %n: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %empty = arith.cmpi eq, %n, %c0_i32 : i32
  %selected = scf.if %empty -> (i32) {
    scf.yield %c0_i32 : i32
  } else {
    %scan:2 = scf.while (%i = %c0_i32) : (i32) -> (i32, i32) {
      %offset = llvm.zext %i : i32 to i64
      %ptr = llvm.getelementptr %src[%offset]
          : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
      %value = llvm.load %ptr : !llvm.ptr -> i32
      %is_match = arith.cmpi eq, %value, %c0_i32 : i32
      %inc = arith.addi %i, %c1_i32 : i32
      %in_bounds = arith.cmpi ult, %i, %n : i32
      %keep_scanning = arith.andi %in_bounds, %is_match : i1
      %next_i = arith.select %is_match, %n, %inc : i32
      scf.condition(%keep_scanning) %next_i, %value : i32, i32
    } do {
    ^bb0(%i_next: i32, %unused: i32):
      scf.yield %i_next : i32
    }
    scf.yield %scan#1 : i32
  }
  llvm.store %selected, %dst : i32, !llvm.ptr
  return
}

// CHECK-LABEL: dataflow.graph.func private @g_standalone_memcpy_0
// CHECK-SAME: (%arg0: none, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32) -> none
// CHECK: llvm.intr.memcpy
// CHECK: dataflow.graph.return %arg0 : none
// CHECK-LABEL: dataflow.graph.func private @g_standalone_offset_memcpy_0
// CHECK-SAME: (%arg0: none, %arg1: !llvm.ptr, %arg2: i16, %arg3: i16, %arg4: i16, %arg5: !llvm.ptr, %arg6: i32) -> none
// CHECK: llvm.zext
// CHECK: arith.muli
// CHECK: llvm.getelementptr
// CHECK: llvm.intr.memcpy
// CHECK: dataflow.graph.return %arg0 : none
// CHECK-LABEL: dataflow.graph.func private @g_structured_outparam_candidate_0
// CHECK-SAME: (%arg0: none, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32) -> none
// CHECK: scf.if
// CHECK: scf.while
// CHECK: llvm.load
// CHECK: llvm.store
// CHECK: dataflow.graph.return %arg0 : none
// CHECK-LABEL: dataflow.graph.func private @g_scatter_add_candidate_0
// CHECK-SAME: (%arg0: none, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) -> none
// CHECK: scf.for
// CHECK: scf.if
// CHECK: llvm.load
// CHECK: llvm.store
// CHECK: dataflow.graph.return %arg0 : none
// CHECK-LABEL: dataflow.graph.func private @g_guarded_loop_candidate_0
// CHECK-SAME: (%arg0: none, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> none
// CHECK: scf.if
// CHECK: scf.for
// CHECK: llvm.load
// CHECK: llvm.store
// CHECK: dataflow.graph.return %arg0 : none
// CHECK-LABEL: dataflow.graph.func private @g_t_unused_ptr_walk_0
// CHECK-SAME: -> none
// CHECK: dataflow.graph.return %{{.*}} : none
// CHECK: dataflow.graph.func private @g_t_host_reduction_red_0_0
// CHECK-SAME: -> (none, f32)
// CHECK: dataflow.graph.return %{{.*}}, %{{.*}} : none, f32
// CHECK-LABEL: dataflow.graph.func private @g_t_forall_store_0
// CHECK: scf.forall
// CHECK: memref.load
// CHECK: memref.store
// CHECK: dataflow.graph.return
// CHECK-LABEL: dataflow.graph.func private @g_t_structured_while_0
// CHECK: scf.if
// CHECK: scf.while
// CHECK: memref.store
// CHECK: dataflow.graph.return
// CHECK-LABEL: dataflow.graph.func private @g_t_structured_index_switch_0
// CHECK: scf.index_switch
// CHECK: memref.store
// CHECK: dataflow.graph.return
