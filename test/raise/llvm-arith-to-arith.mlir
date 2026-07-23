// RUN: loom-raise-opt --loom-llvm-func-to-func --loom-llvm-arith-to-arith %s | FileCheck %s

// Verify that LLVM arithmetic, compare and constant ops with builtin
// numeric types are rewritten into the matching arith dialect ops, and
// that every semantic flag arith cannot express keeps its op in llvm form.
// Pointer-typed ops (gep/load/store/alloca) and llvm.* float-ext / cast
// ops with exotic types stay in the llvm dialect on purpose.
//
// The arith-to-arith pass is nested under func.func, so llvm.func inputs
// are first raised via loom-llvm-func-to-func. Cases whose subject is a
// signature func-to-func does not raise -- vector element types -- or whose
// values are only observable as several results state their container as a
// func.func directly, so what they anchor is the arith rewrite alone.

// CHECK-LABEL: func.func @int_arith
llvm.func @int_arith(%a: i32, %b: i32) -> i32 {
    // CHECK: %{{.*}} = arith.addi %arg0, %arg1
    %0 = llvm.add %a, %b : i32
    // CHECK: %{{.*}} = arith.muli
    %1 = llvm.mul %0, %a : i32
    // CHECK: %{{.*}} = arith.subi
    %2 = llvm.sub %1, %b : i32
    llvm.return %2 : i32
}

// CHECK-LABEL: func.func @float_arith
llvm.func @float_arith(%a: f32, %b: f32) -> f32 {
    // CHECK: %{{.*}} = arith.addf %arg0, %arg1
    %0 = llvm.fadd %a, %b : f32
    // CHECK: %{{.*}} = arith.mulf
    %1 = llvm.fmul %0, %a : f32
    llvm.return %1 : f32
}

// Integer overflow flags are exactly representable in arith and must
// survive the rewrite.
// CHECK-LABEL: func.func @int_overflow_flags
llvm.func @int_overflow_flags(%a: i32, %b: i32) -> i32 {
    // CHECK: %{{.*}} = arith.addi %arg0, %arg1 overflow<nsw> : i32
    %0 = llvm.add %a, %b overflow<nsw> : i32
    // CHECK: %{{.*}} = arith.shli %{{.*}}, %arg1 overflow<nuw> : i32
    %1 = llvm.shl %0, %b overflow<nuw> : i32
    // CHECK: %{{.*}} = arith.muli %{{.*}}, %arg0 overflow<nsw, nuw> : i32
    %2 = llvm.mul %1, %a overflow<nsw, nuw> : i32
    llvm.return %2 : i32
}

// The exact flag is exactly representable on the matching arith ops.
// CHECK-LABEL: func.func @int_exact_flag
llvm.func @int_exact_flag(%a: i32, %b: i32) -> i32 {
    // CHECK: %{{.*}} = arith.divsi %arg0, %arg1 exact : i32
    %0 = llvm.sdiv exact %a, %b : i32
    // CHECK: %{{.*}} = arith.shrui %{{.*}}, %arg1 exact : i32
    %1 = llvm.lshr exact %0, %b : i32
    llvm.return %1 : i32
}

// LLVM and arith fast-math flags name the same facts but use different
// bit positions, so each flag must be mapped by name.
// CHECK-LABEL: func.func @float_fastmath_flags
llvm.func @float_fastmath_flags(%a: f32, %b: f32) -> f32 {
    // CHECK: %{{.*}} = arith.addf %arg0, %arg1 fastmath<nnan> : f32
    %0 = llvm.fadd %a, %b {fastmathFlags = #llvm.fastmath<nnan>} : f32
    // CHECK: %{{.*}} = arith.mulf %{{.*}}, %arg0 fastmath<contract> : f32
    %1 = llvm.fmul %0, %a {fastmathFlags = #llvm.fastmath<contract>} : f32
    // CHECK: %{{.*}} = arith.subf %{{.*}}, %arg1 fastmath<ninf,nsz,arcp,afn> : f32
    %2 = llvm.fsub %1, %b {fastmathFlags = #llvm.fastmath<ninf, nsz, arcp, afn>} : f32
    // CHECK: %{{.*}} = arith.divf %{{.*}}, %arg1 fastmath<fast> : f32
    %3 = llvm.fdiv %2, %b {fastmathFlags = #llvm.fastmath<fast>} : f32
    llvm.return %3 : f32
}

// arith.cmpf carries fast-math flags; arith.select does not, so a
// flagged llvm.select has to stay in llvm form.
// CHECK-LABEL: func.func @float_fastmath_users
llvm.func @float_fastmath_users(%a: f32, %b: f32) -> f32 {
    // CHECK: %{{.*}} = arith.cmpf oeq, %arg0, %arg1 fastmath<nnan> : f32
    %0 = llvm.fcmp "oeq" %a, %b {fastmathFlags = #llvm.fastmath<nnan>} : f32
    // CHECK: %{{.*}} = llvm.select {{.*}}fastmath<contract>
    // CHECK-NOT: arith.select
    %1 = llvm.select %0, %a, %b {fastmathFlags = #llvm.fastmath<contract>} : i1, f32
    llvm.return %1 : f32
}

// llvm.or's disjoint flag has no arith counterpart, so that op stays in
// llvm form while its exactly representable neighbour is raised.
// CHECK-LABEL: func.func @or_disjoint_stays_llvm
llvm.func @or_disjoint_stays_llvm(%a: i32, %b: i32) -> i32 {
    // CHECK: %{{.*}} = llvm.or disjoint %arg0, %arg1 : i32
    %0 = llvm.or disjoint %a, %b : i32
    // CHECK: %{{.*}} = arith.ori %{{.*}}, %arg1 : i32
    %1 = llvm.or %0, %b : i32
    llvm.return %1 : i32
}

// A fixed-shape vector of a builtin element type is exactly what arith
// consumes, so shape, element semantics and flags all survive.
// CHECK-LABEL: func.func @fixed_vector_arith
func.func @fixed_vector_arith(%a: vector<4xi32>, %b: vector<4xi32>) -> vector<4xi32> {
    // CHECK: %{{.*}} = arith.addi %arg0, %arg1 overflow<nsw> : vector<4xi32>
    %0 = llvm.add %a, %b overflow<nsw> : vector<4xi32>
    // CHECK: %{{.*}} = arith.muli %{{.*}}, %arg0 : vector<4xi32>
    %1 = llvm.mul %0, %a : vector<4xi32>
    return %1 : vector<4xi32>
}

// CHECK-LABEL: func.func @fixed_vector_float_arith
func.func @fixed_vector_float_arith(%a: vector<4xf32>, %b: vector<4xf32>) -> (vector<4xf32>, vector<4xi1>) {
    // CHECK: %{{.*}} = arith.addf %arg0, %arg1 fastmath<nnan> : vector<4xf32>
    %0 = llvm.fadd %a, %b {fastmathFlags = #llvm.fastmath<nnan>} : vector<4xf32>
    // CHECK: %{{.*}} = arith.cmpf olt, %{{.*}}, %arg1 : vector<4xf32>
    %1 = llvm.fcmp "olt" %0, %b : vector<4xf32>
    return %0, %1 : vector<4xf32>, vector<4xi1>
}

// A scalable vector's length is a runtime fact, and no authority defines
// what raising one means, so its ops stay in llvm form.
// CHECK-LABEL: func.func @scalable_vector_stays_llvm
func.func @scalable_vector_stays_llvm(%a: vector<[4]xi32>, %b: vector<[4]xi32>) -> (vector<[4]xi32>, vector<[4]xi1>) {
    // CHECK: %{{.*}} = llvm.add %arg0, %arg1 : vector<[4]xi32>
    // CHECK-NOT: arith.addi
    %0 = llvm.add %a, %b : vector<[4]xi32>
    // CHECK: %{{.*}} = llvm.icmp "slt" %{{.*}}, %arg1 : vector<[4]xi32>
    // CHECK-NOT: arith.cmpi
    %1 = llvm.icmp "slt" %0, %b : vector<[4]xi32>
    return %0, %1 : vector<[4]xi32>, vector<[4]xi1>
}

// Every predicate the pinned LLVM dialect defines is mapped by name, so a
// reordering of either upstream enum cannot silently change what a
// comparison means. Returning the results keeps each one observable
// without a sink symbol.
// CHECK-LABEL: func.func @int_cmp
func.func @int_cmp(%a: i32, %b: i32) -> (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1) {
    // CHECK: arith.cmpi eq,
    %0 = llvm.icmp "eq" %a, %b : i32
    // CHECK: arith.cmpi ne,
    %1 = llvm.icmp "ne" %a, %b : i32
    // CHECK: arith.cmpi slt,
    %2 = llvm.icmp "slt" %a, %b : i32
    // CHECK: arith.cmpi sle,
    %3 = llvm.icmp "sle" %a, %b : i32
    // CHECK: arith.cmpi sgt,
    %4 = llvm.icmp "sgt" %a, %b : i32
    // CHECK: arith.cmpi sge,
    %5 = llvm.icmp "sge" %a, %b : i32
    // CHECK: arith.cmpi ult,
    %6 = llvm.icmp "ult" %a, %b : i32
    // CHECK: arith.cmpi ule,
    %7 = llvm.icmp "ule" %a, %b : i32
    // CHECK: arith.cmpi ugt,
    %8 = llvm.icmp "ugt" %a, %b : i32
    // CHECK: arith.cmpi uge,
    %9 = llvm.icmp "uge" %a, %b : i32
    // CHECK-NOT: llvm.icmp
    return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: func.func @float_cmp
func.func @float_cmp(%a: f32, %b: f32) -> (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1) {
    // CHECK: arith.cmpf false,
    %0 = llvm.fcmp "_false" %a, %b : f32
    // CHECK: arith.cmpf oeq,
    %1 = llvm.fcmp "oeq" %a, %b : f32
    // CHECK: arith.cmpf ogt,
    %2 = llvm.fcmp "ogt" %a, %b : f32
    // CHECK: arith.cmpf oge,
    %3 = llvm.fcmp "oge" %a, %b : f32
    // CHECK: arith.cmpf olt,
    %4 = llvm.fcmp "olt" %a, %b : f32
    // CHECK: arith.cmpf ole,
    %5 = llvm.fcmp "ole" %a, %b : f32
    // CHECK: arith.cmpf one,
    %6 = llvm.fcmp "one" %a, %b : f32
    // CHECK: arith.cmpf ord,
    %7 = llvm.fcmp "ord" %a, %b : f32
    // CHECK: arith.cmpf ueq,
    %8 = llvm.fcmp "ueq" %a, %b : f32
    // CHECK: arith.cmpf ugt,
    %9 = llvm.fcmp "ugt" %a, %b : f32
    // CHECK: arith.cmpf uge,
    %10 = llvm.fcmp "uge" %a, %b : f32
    // CHECK: arith.cmpf ult,
    %11 = llvm.fcmp "ult" %a, %b : f32
    // CHECK: arith.cmpf ule,
    %12 = llvm.fcmp "ule" %a, %b : f32
    // CHECK: arith.cmpf une,
    %13 = llvm.fcmp "une" %a, %b : f32
    // CHECK: arith.cmpf uno,
    %14 = llvm.fcmp "uno" %a, %b : f32
    // CHECK: arith.cmpf true,
    %15 = llvm.fcmp "_true" %a, %b : f32
    // CHECK-NOT: llvm.fcmp
    return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: func.func @numeric_select
llvm.func @numeric_select(%cond: i1, %a: i32, %b: i32) -> i32 {
    // CHECK: %{{.*}} = arith.select %arg0, %arg1, %arg2 : i32
    %0 = llvm.select %cond, %a, %b : i1, i32
    llvm.return %0 : i32
}

// CHECK-LABEL: func.func @int_constant
llvm.func @int_constant() -> i32 {
    // CHECK: %{{.*}} = arith.constant 42 : i32
    %0 = llvm.mlir.constant(42 : i32) : i32
    llvm.return %0 : i32
}

// CHECK-LABEL: func.func @ptr_load_stays_llvm
// Load of a pointer-typed memory remains as llvm.load -- we only
// rewrite arith and comparison ops here.
llvm.func @ptr_load_stays_llvm(%p: !llvm.ptr) -> f32 {
    // CHECK: %{{.*}} = llvm.load
    %v = llvm.load %p : !llvm.ptr -> f32
    llvm.return %v : f32
}

// CHECK-LABEL: func.func @ptr_select_stays_llvm
llvm.func @ptr_select_stays_llvm(%cond: i1, %a: !llvm.ptr, %b: !llvm.ptr) -> !llvm.ptr {
    // CHECK: %{{.*}} = llvm.select
    %0 = llvm.select %cond, %a, %b : i1, !llvm.ptr
    llvm.return %0 : !llvm.ptr
}

llvm.func @expf(f32) -> f32

// A recognized libm symbol proves nothing a pure math op needs. An absent
// memory-effects attribute is LLVM's default read/write set, not a promise
// of purity, and `nobuiltin` states outright that the caller did not ask
// for the builtin -- the spelling of the name settles neither errno, the
// FP environment, nor termination. Both calls are left to a source form
// that owns that contract.
// CHECK-LABEL: func.func @libm_calls_stay_llvm
func.func @libm_calls_stay_llvm(%x: f32) -> (f32, f32) {
    // CHECK: %{{.*}} = llvm.call @expf(%arg0) : (f32) -> f32
    %0 = llvm.call @expf(%x) : (f32) -> f32
    // CHECK: %{{.*}} = llvm.call @expf(%arg0) {nobuiltin} : (f32) -> f32
    %1 = llvm.call @expf(%x) {nobuiltin} : (f32) -> f32
    // CHECK-NOT: math.exp
    return %0, %1 : f32, f32
}
