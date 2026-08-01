// RUN: loom-raise-opt --loom-llvm-arith-to-arith %s | FileCheck %s

// Verify that each LLVM computation whose complete semantics an arith or math
// operation restates exactly is rewritten into that standard operation, and
// that every source fact the standard operation cannot carry keeps its
// operation in llvm form. Pointer-typed ops (gep/load/store/alloca) stay in
// the llvm dialect on purpose.
//
// The pass rewrites every callable region in place, so an imported llvm.func
// is normalized where it stands and stays the sole owner of its ABI envelope.
// A case whose values are only observable as several results states its
// container as a func.func, which anchors the same rewrite on the other
// callable kind.

// CHECK-LABEL: llvm.func @int_arith
llvm.func @int_arith(%a: i32, %b: i32) -> i32 {
    // CHECK: %{{.*}} = arith.addi %arg0, %arg1
    %0 = llvm.add %a, %b : i32
    // CHECK: %{{.*}} = arith.muli
    %1 = llvm.mul %0, %a : i32
    // CHECK: %{{.*}} = arith.subi
    %2 = llvm.sub %1, %b : i32
    llvm.return %2 : i32
}

// CHECK-LABEL: llvm.func @float_arith
llvm.func @float_arith(%a: f32, %b: f32) -> f32 {
    // CHECK: %{{.*}} = arith.addf %arg0, %arg1
    %0 = llvm.fadd %a, %b : f32
    // CHECK: %{{.*}} = arith.mulf
    %1 = llvm.fmul %0, %a : f32
    llvm.return %1 : f32
}

// Integer overflow flags are exactly representable in arith and must
// survive the rewrite.
// CHECK-LABEL: llvm.func @int_overflow_flags
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
// CHECK-LABEL: llvm.func @int_exact_flag
llvm.func @int_exact_flag(%a: i32, %b: i32) -> i32 {
    // CHECK: %{{.*}} = arith.divsi %arg0, %arg1 exact : i32
    %0 = llvm.sdiv exact %a, %b : i32
    // CHECK: %{{.*}} = arith.shrui %{{.*}}, %arg1 exact : i32
    %1 = llvm.lshr exact %0, %b : i32
    llvm.return %1 : i32
}

// LLVM and arith fast-math flags name the same facts but use different
// bit positions, so each flag must be mapped by name.
// CHECK-LABEL: llvm.func @float_fastmath_flags
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
// CHECK-LABEL: llvm.func @float_fastmath_users
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
// CHECK-LABEL: llvm.func @or_disjoint_stays_llvm
llvm.func @or_disjoint_stays_llvm(%a: i32, %b: i32) -> i32 {
    // CHECK: %{{.*}} = llvm.or disjoint %arg0, %arg1 : i32
    %0 = llvm.or disjoint %a, %b : i32
    // CHECK: %{{.*}} = arith.ori %{{.*}}, %arg1 : i32
    %1 = llvm.or %0, %b : i32
    llvm.return %1 : i32
}

// A width or domain cast states which bits the result keeps, how the source
// pattern is read, and how the result rounds. The signedness reading is part
// of the operation identity on both sides, so each source cast reaches its own
// standard counterpart rather than a shared one.
// CHECK-LABEL: llvm.func @width_and_domain_casts
llvm.func @width_and_domain_casts(%narrow: i16, %wide: i32, %single: f32,
                                  %double: f64) {
    // CHECK: %{{.*}} = arith.trunci %arg1 : i32 to i16
    %0 = llvm.trunc %wide : i32 to i16
    // CHECK: %{{.*}} = arith.extui %arg0 : i16 to i32
    %1 = llvm.zext %narrow : i16 to i32
    // CHECK: %{{.*}} = arith.extsi %arg0 : i16 to i32
    %2 = llvm.sext %narrow : i16 to i32
    // CHECK: %{{.*}} = arith.sitofp %arg1 : i32 to f32
    %3 = llvm.sitofp %wide : i32 to f32
    // CHECK: %{{.*}} = arith.uitofp %arg1 : i32 to f32
    %4 = llvm.uitofp %wide : i32 to f32
    // CHECK: %{{.*}} = arith.fptosi %arg2 : f32 to i32
    %5 = llvm.fptosi %single : f32 to i32
    // CHECK: %{{.*}} = arith.fptoui %arg2 : f32 to i32
    %6 = llvm.fptoui %single : f32 to i32
    // CHECK: %{{.*}} = arith.extf %arg2 : f32 to f64
    %7 = llvm.fpext %single : f32 to f64
    // CHECK: %{{.*}} = arith.truncf %arg3 : f64 to f32
    %8 = llvm.fptrunc %double : f64 to f32
    // CHECK-NOT: llvm.trunc
    // CHECK-NOT: llvm.zext
    llvm.return
}

// A scalar LLVM bitcast and an elementwise fixed-vector bitcast have the exact
// standard spelling. An equal-total-width cast that changes vector shape does
// not: arith.bitcast is elementwise, so that LLVM operation remains explicit.
// CHECK-LABEL: llvm.func @bit_reinterpretation
llvm.func @bit_reinterpretation(%bits: i16, %lanes: vector<2xi16>,
                                %packed: vector<2xi8>) {
    // CHECK: %{{.*}} = arith.bitcast %arg0 : i16 to f16
    %0 = llvm.bitcast %bits : i16 to f16
    // CHECK: %{{.*}} = arith.bitcast %arg1 : vector<2xi16> to vector<2xf16>
    %1 = llvm.bitcast %lanes : vector<2xi16> to vector<2xf16>
    // CHECK: %{{.*}} = llvm.bitcast %arg2 : vector<2xi8> to i16
    %2 = llvm.bitcast %packed : vector<2xi8> to i16
    llvm.return
}

// Neither narrowing cast acquires a rounding mode: an arith cast that states
// one is a constrained operation, while the source is an ordinary cast in the
// default floating-point environment. Every source flag that has an exact and
// roundtrippable carrier is transferred: nsw / nuw on truncation and nneg on
// the two casts that read the operand as unsigned. A fast-math contract on a
// floating resize is the one exception: the pinned arith-to-llvm lowering of a
// fast-math arith.extf or arith.truncf does not carry it back onto the llvm
// op, so a flagged resize keeps its llvm form rather than break the round trip.
// CHECK-LABEL: llvm.func @cast_flags
llvm.func @cast_flags(%narrow: i16, %wide: i32, %single: f32) {
    // CHECK: %{{.*}} = arith.trunci %arg1 overflow<nsw, nuw> : i32 to i16
    %0 = llvm.trunc %wide overflow<nsw, nuw> : i32 to i16
    // CHECK: %{{.*}} = arith.extui %arg0 nneg : i16 to i32
    %1 = llvm.zext nneg %narrow : i16 to i32
    // CHECK: %{{.*}} = arith.uitofp %arg1 nneg : i32 to f32
    %2 = llvm.uitofp nneg %wide : i32 to f32
    // CHECK: %{{.*}} = llvm.fpext %arg2 fastmath<nnan> : f32 to f64
    %3 = llvm.fpext %single fastmath<nnan> : f32 to f64
    llvm.return
}

// Sign flip and magnitude, and the four minimum/maximum families. The two
// floating families differ in what they state about NaN and signed zero, so
// each keeps its own identity instead of collapsing into one spelling.
// CHECK-LABEL: llvm.func @unary_and_extremum
llvm.func @unary_and_extremum(%i: i32, %f: f32) {
    // CHECK: %{{.*}} = arith.negf %arg1 : f32
    %0 = llvm.fneg %f : f32
    // CHECK: %{{.*}} = math.absf %arg1 fastmath<ninf> : f32
    %1 = llvm.intr.fabs(%f) {fastmathFlags = #llvm.fastmath<ninf>} : (f32) -> f32
    // CHECK: %{{.*}} = arith.maxnumf %arg1, %arg1 : f32
    %2 = llvm.intr.maxnum(%f, %f) : (f32, f32) -> f32
    // CHECK: %{{.*}} = arith.minnumf %arg1, %arg1 : f32
    %3 = llvm.intr.minnum(%f, %f) : (f32, f32) -> f32
    // CHECK: %{{.*}} = arith.maximumf %arg1, %arg1 : f32
    %4 = llvm.intr.maximum(%f, %f) : (f32, f32) -> f32
    // CHECK: %{{.*}} = arith.minimumf %arg1, %arg1 : f32
    %5 = llvm.intr.minimum(%f, %f) : (f32, f32) -> f32
    // CHECK: %{{.*}} = arith.maxsi %arg0, %arg0 : i32
    %6 = llvm.intr.smax(%i, %i) : (i32, i32) -> i32
    // CHECK: %{{.*}} = arith.minsi %arg0, %arg0 : i32
    %7 = llvm.intr.smin(%i, %i) : (i32, i32) -> i32
    // CHECK: %{{.*}} = arith.maxui %arg0, %arg0 : i32
    %8 = llvm.intr.umax(%i, %i) : (i32, i32) -> i32
    // CHECK: %{{.*}} = arith.minui %arg0, %arg0 : i32
    %9 = llvm.intr.umin(%i, %i) : (i32, i32) -> i32
    llvm.return
}

// llvm.intr.fma is the exact fused multiply-add, so it has one standard
// counterpart. llvm.intr.fmuladd states a choice between that fused form and
// a separate multiply and add, which no single standard operation restates,
// so it survives mechanical raising untouched.
// CHECK-LABEL: llvm.func @fused_multiply_add
llvm.func @fused_multiply_add(%f: f32) {
    // CHECK: %{{.*}} = math.fma %arg0, %arg0, %arg0 fastmath<contract> : f32
    %0 = llvm.intr.fma(%f, %f, %f)
        {fastmathFlags = #llvm.fastmath<contract>} : (f32, f32, f32) -> f32
    // CHECK: %{{.*}} = llvm.intr.fmuladd(%arg0, %arg0, %arg0)
    // CHECK-NOT: math.fma
    %1 = llvm.intr.fmuladd(%f, %f, %f) : (f32, f32, f32) -> f32
    llvm.return
}

// A frontend that has explicitly removed the libm errno contract emits the
// typed LLVM cosine intrinsic. The standard math operation carries the same
// operand, result, and fast-math contract without consulting a symbol name.
// CHECK-LABEL: llvm.func @typed_cosine
llvm.func @typed_cosine(%single: f32, %double: f64) {
    // CHECK: %{{.*}} = math.cos %arg0 : f32
    %0 = llvm.intr.cos(%single) : (f32) -> f32
    // CHECK: %{{.*}} = math.cos %arg1 fastmath<afn> : f64
    %1 = llvm.intr.cos(%double) {fastmathFlags = #llvm.fastmath<afn>} : (f64) -> f64
    // CHECK-NOT: llvm.intr.cos
    llvm.return
}

// Typed LLVM elementary-math intrinsics already state the complete operation
// identity and floating-point contract. They therefore use the existing math
// schemas rather than surviving as LLVM aliases in a selected graph.
// CHECK-LABEL: llvm.func @typed_elementary_math
llvm.func @typed_elementary_math(%single: f32, %double: f64) {
    // CHECK: %{{.*}} = math.sqrt %arg0 : f32
    %0 = llvm.intr.sqrt(%single) : (f32) -> f32
    // CHECK: %{{.*}} = math.exp %arg0 fastmath<afn> : f32
    %1 = llvm.intr.exp(%single) {fastmathFlags = #llvm.fastmath<afn>} : (f32) -> f32
    // CHECK: %{{.*}} = math.log %arg1 : f64
    %2 = llvm.intr.log(%double) : (f64) -> f64
    // CHECK: %{{.*}} = math.ceil %arg0 : f32
    %3 = llvm.intr.ceil(%single) : (f32) -> f32
    // CHECK: %{{.*}} = math.floor %arg0 : f32
    %4 = llvm.intr.floor(%single) : (f32) -> f32
    // CHECK: %{{.*}} = math.trunc %arg1 : f64
    %5 = llvm.intr.trunc(%double) : (f64) -> f64
    // CHECK: %{{.*}} = math.powf %arg0, %arg0 fastmath<afn> : f32
    %6 = llvm.intr.pow(%single, %single) {fastmathFlags = #llvm.fastmath<afn>} : (f32, f32) -> f32
    // CHECK-NOT: llvm.intr.sqrt
    // CHECK-NOT: llvm.intr.exp
    // CHECK-NOT: llvm.intr.log
    // CHECK-NOT: llvm.intr.ceil
    // CHECK-NOT: llvm.intr.floor
    // CHECK-NOT: llvm.intr.trunc
    // CHECK-NOT: llvm.intr.pow
    llvm.return
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

// CHECK-LABEL: func.func @fixed_vector_casts_and_extremum
func.func @fixed_vector_casts_and_extremum(%i: vector<4xi16>, %f: vector<4xf32>)
        -> (vector<4xi32>, vector<4xf64>, vector<4xf32>) {
    // CHECK: %{{.*}} = arith.extsi %arg0 : vector<4xi16> to vector<4xi32>
    %0 = llvm.sext %i : vector<4xi16> to vector<4xi32>
    // CHECK: %{{.*}} = arith.extf %arg1 : vector<4xf32> to vector<4xf64>
    %1 = llvm.fpext %f : vector<4xf32> to vector<4xf64>
    // CHECK: %{{.*}} = arith.maxnumf %arg1, %arg1 : vector<4xf32>
    %2 = llvm.intr.maxnum(%f, %f) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    return %0, %1, %2 : vector<4xi32>, vector<4xf64>, vector<4xf32>
}

// A scalable vector's element count is a runtime `vscale` multiple rather
// than a shape, so every alias fails closed on one. Its operations keep their
// llvm form until a typed structured transform has materialized the
// computation as fixed-width chunks, loops, and masks or tails.
// CHECK-LABEL: func.func @scalable_vector_stays_llvm
func.func @scalable_vector_stays_llvm(%a: vector<[4]xi32>, %b: vector<[4]xi32>,
                                      %f: vector<[4]xf32>)
        -> (vector<[4]xi32>, vector<[4]xi1>, vector<[4]xi64>, vector<[4]xf32>) {
    // CHECK: %{{.*}} = llvm.add %arg0, %arg1 : vector<[4]xi32>
    // CHECK-NOT: arith.addi
    %0 = llvm.add %a, %b : vector<[4]xi32>
    // CHECK: %{{.*}} = llvm.icmp "slt" %{{.*}}, %arg1 : vector<[4]xi32>
    // CHECK-NOT: arith.cmpi
    %1 = llvm.icmp "slt" %0, %b : vector<[4]xi32>
    // CHECK: %{{.*}} = llvm.zext %arg0 : vector<[4]xi32> to vector<[4]xi64>
    // CHECK-NOT: arith.extui
    %2 = llvm.zext %a : vector<[4]xi32> to vector<[4]xi64>
    // CHECK: %{{.*}} = llvm.intr.maxnum(%arg2, %arg2)
    // CHECK-NOT: arith.maxnumf
    %3 = llvm.intr.maxnum(%f, %f) : (vector<[4]xf32>, vector<[4]xf32>) -> vector<[4]xf32>
    return %0, %1, %2, %3 : vector<[4]xi32>, vector<[4]xi1>, vector<[4]xi64>, vector<[4]xf32>
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

// CHECK-LABEL: llvm.func @numeric_select
llvm.func @numeric_select(%cond: i1, %a: i32, %b: i32) -> i32 {
    // CHECK: %{{.*}} = arith.select %arg0, %arg1, %arg2 : i32
    %0 = llvm.select %cond, %a, %b : i1, i32
    llvm.return %0 : i32
}

// CHECK-LABEL: llvm.func @zero_count_aliases
llvm.func @zero_count_aliases(%value: i32) -> i32 {
    // CHECK: %[[CTLZ:.*]] = math.ctlz %arg0 : i32
    %ctlz = "llvm.intr.ctlz"(%value) <{is_zero_poison = false}> : (i32) -> i32
    // CHECK: %[[CTTZ:.*]] = math.cttz %arg0 : i32
    %cttz = "llvm.intr.cttz"(%value) <{is_zero_poison = false}> : (i32) -> i32
    // CHECK: %[[CTLZ_POISON:.*]] = "llvm.intr.ctlz"(%arg0) <{is_zero_poison = true}> : (i32) -> i32
    %ctlz_poison = "llvm.intr.ctlz"(%value) <{is_zero_poison = true}> : (i32) -> i32
    // CHECK: %[[CTTZ_POISON:.*]] = "llvm.intr.cttz"(%arg0) <{is_zero_poison = true}> : (i32) -> i32
    %cttz_poison = "llvm.intr.cttz"(%value) <{is_zero_poison = true}> : (i32) -> i32
    llvm.return %ctlz : i32
}

// Integer absolute value is not a distinct builtin Fabric capability. The
// mechanical S0 spelling uses ordinary compare, subtract, and select actors.
// The poisoning LLVM form carries its INT_MIN contract on the negation.
// CHECK-LABEL: llvm.func @integer_abs_defined
llvm.func @integer_abs_defined(%value: i32) -> i32 {
    // CHECK: %[[ZERO0:.*]] = arith.constant 0 : i32
    // CHECK: %[[NEG0:.*]] = arith.cmpi slt, %arg0, %[[ZERO0]] : i32
    // CHECK: %[[MAG0:.*]] = arith.subi %[[ZERO0]], %arg0 : i32
    // CHECK: %[[ABS0:.*]] = arith.select %[[NEG0]], %[[MAG0]], %arg0 : i32
    %defined = "llvm.intr.abs"(%value) <{is_int_min_poison = false}> : (i32) -> i32
    llvm.return %defined : i32
}

// CHECK-LABEL: llvm.func @integer_abs_poison
llvm.func @integer_abs_poison(%value: i32) -> i32 {
    // CHECK: %[[ZERO1:.*]] = arith.constant 0 : i32
    // CHECK: %[[NEG1:.*]] = arith.cmpi slt, %arg0, %[[ZERO1]] : i32
    // CHECK: %[[MAG1:.*]] = arith.subi %[[ZERO1]], %arg0 overflow<nsw> : i32
    // CHECK: %[[ABS1:.*]] = arith.select %[[NEG1]], %[[MAG1]], %arg0 : i32
    // CHECK-NOT: llvm.intr.abs
    %poison = "llvm.intr.abs"(%value) <{is_int_min_poison = true}> : (i32) -> i32
    llvm.return %poison : i32
}

// CHECK-LABEL: llvm.func @int_constant
llvm.func @int_constant() -> i32 {
    // CHECK: %{{.*}} = arith.constant 42 : i32
    %0 = llvm.mlir.constant(42 : i32) : i32
    llvm.return %0 : i32
}

// CHECK-LABEL: llvm.func @ptr_load_stays_llvm
// Load of a pointer-typed memory remains as llvm.load -- we only
// rewrite arith and comparison ops here.
llvm.func @ptr_load_stays_llvm(%p: !llvm.ptr) -> f32 {
    // CHECK: %{{.*}} = llvm.load
    %v = llvm.load %p : !llvm.ptr -> f32
    llvm.return %v : f32
}

// CHECK-LABEL: llvm.func @ptr_select_stays_llvm
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
