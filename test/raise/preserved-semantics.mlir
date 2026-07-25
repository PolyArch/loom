// RUN: split-file %s %t
// RUN: %loom-raise %t/frozen-undef.ll | FileCheck %s --check-prefix=EXCEPTIONAL
// RUN: loom-raise-opt --loom-llvm-arith-to-arith %t/environment.mlir | FileCheck %s --check-prefix=ENVIRONMENT
// RUN: loom-raise-opt --mlir-print-debuginfo --loom-llvm-arith-to-arith %t/provenance.mlir | FileCheck %s --check-prefix=PROVENANCE

// Mechanical raising preserves what it cannot restate exactly, and carries
// what it can onto the operation that replaces the source.

// Neither undef nor freeze has an exact standard normalization, so both
// survive structuring unchanged. Replacing the undef with an ordinary constant
// would make the freeze observe a chosen value.
// EXCEPTIONAL-LABEL: llvm.func @frozen_undef
// EXCEPTIONAL: %[[UNDEF:.*]] = llvm.mlir.undef : i32
// EXCEPTIONAL: scf.if
// EXCEPTIONAL: llvm.freeze %[[UNDEF]]
// EXCEPTIONAL: llvm.return

// An unflagged standard floating operation means the pinned default LLVM
// floating-point environment, and neither arith nor math states an enclosing
// environment of its own. A callable stating the default environment is
// normalized; one stating anything else keeps its floating operations in llvm
// form rather than having them silently reinterpreted.
//
// A reciprocal-estimate policy names the operations the target may compute as
// an estimate plus refinement rather than exactly, a non-IEEE denormal mode
// and a nonempty fp-contract permission both change the result, and any other
// environment fact arrives through the opaque passthrough collection.
// Integer arithmetic is independent of that environment and is normalized
// beside a blocked floating operation.
// ENVIRONMENT-LABEL: llvm.func @default_environment
// ENVIRONMENT: arith.addf
// ENVIRONMENT-LABEL: llvm.func @estimated
// ENVIRONMENT: llvm.fadd
// ENVIRONMENT-LABEL: llvm.func @flushed
// ENVIRONMENT: llvm.fadd
// ENVIRONMENT: arith.addi
// ENVIRONMENT-LABEL: llvm.func @contracted
// ENVIRONMENT: llvm.fmul
// ENVIRONMENT-LABEL: llvm.func @target_specific
// ENVIRONMENT: llvm.fadd
// ENVIRONMENT-NOT: arith.addf
// ENVIRONMENT-NOT: arith.mulf

// The imported debug scope and the source file, line and column of the
// replaced operation are the provenance authority, so the standard operation
// that replaces it carries exactly the same location.
// PROVENANCE-LABEL: llvm.func @carried
// PROVENANCE: arith.addi {{.*}} loc(#[[BODY:loc[0-9]+]])
// PROVENANCE-DAG: #[[SCOPE:.*]] = #llvm.di_subprogram<{{.*}}name = "carried"
// PROVENANCE-DAG: #[[ADDS:loc[0-9]+]] = loc("source.c":9:3)
// PROVENANCE-DAG: #[[BODY]] = loc(fused<#[[SCOPE]]>[#[[ADDS]]])

//--- frozen-undef.ll
define i32 @frozen_undef(i1 %select) {
entry:
  br i1 %select, label %frozen, label %raw

frozen:
  %stable = freeze i32 undef
  br label %exit

raw:
  br label %exit

exit:
  %result = phi i32 [ %stable, %frozen ], [ 0, %raw ]
  ret i32 %result
}

//--- environment.mlir
llvm.func @default_environment(%a: f32, %b: f32) -> f32 {
  %0 = llvm.fadd %a, %b : f32
  llvm.return %0 : f32
}

llvm.func @estimated(%a: f32, %b: f32) -> f32
    attributes {reciprocal_estimates = "all"} {
  %0 = llvm.fadd %a, %b : f32
  llvm.return %0 : f32
}

llvm.func @flushed(%a: f32, %b: f32, %i: i32) -> f32
    attributes {denormal_fpenv = #llvm.denormal_fpenv<
        default_output_mode = preservesign, default_input_mode = ieee,
        float_output_mode = ieee, float_input_mode = ieee>} {
  %0 = llvm.fadd %a, %b : f32
  %1 = llvm.add %i, %i : i32
  llvm.return %0 : f32
}

llvm.func @contracted(%a: f32, %b: f32) -> f32
    attributes {fp_contract = "fast"} {
  %0 = llvm.fmul %a, %b : f32
  llvm.return %0 : f32
}

llvm.func @target_specific(%a: f32, %b: f32) -> f32
    attributes {passthrough = ["nounwind"]} {
  %0 = llvm.fadd %a, %b : f32
  llvm.return %0 : f32
}

//--- provenance.mlir
#file = #llvm.di_file<"source.c" in "/src">
#cu = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C99,
                            file = #file, isOptimized = false,
                            emissionKind = Full>
#signature = #llvm.di_subroutine_type<callingConvention = DW_CC_normal>
#program = #llvm.di_subprogram<id = distinct[1]<>, compileUnit = #cu,
                               scope = #file, name = "carried", file = #file,
                               line = 7, subprogramFlags = Definition,
                               type = #signature>
#adds = loc("source.c":9:3)
#body = loc(fused<#program>[#adds])
#entry = loc("source.c":7:1)
#declaration = loc(fused<#program>[#entry])

llvm.func @carried(%a: i32, %b: i32) -> i32 {
  %0 = llvm.add %a, %b : i32 loc(#body)
  llvm.return %0 : i32 loc("source.c":10:3)
} loc(#declaration)
