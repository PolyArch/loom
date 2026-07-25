// RUN: loom-raise-opt --loom-llvm-cf-to-cf --loom-lift-cf-to-scf --loom-llvm-arith-to-arith %s | FileCheck %s --implicit-check-not=func.func

// The imported LLVM function stays the sole callable and ABI owner while its
// CFG is structured and its computations are normalized in place. Linkage,
// calling convention, COMDAT, personality, argument and result attributes,
// memory effects, target features and the floating-point environment all stay
// on the same llvm.func, and nothing is copied into another dialect to obtain
// a pass wrapper.

llvm.comdat @__llvm_comdat {
  llvm.comdat_selector @envelope any
}

llvm.func @__gxx_personality_v0(...) -> i32

// CHECK-LABEL: llvm.func weak_odr fastcc @envelope
// CHECK-SAME: !llvm.ptr {llvm.noalias, llvm.readonly}
// CHECK-SAME: -> (i32 {llvm.signext})
// CHECK-SAME: comdat(@__llvm_comdat::@envelope)
// CHECK-SAME: denormal_fpenv = #llvm.denormal_fpenv<default_output_mode = preservesign,
// CHECK-SAME: fp_contract = "fast"
// CHECK-SAME: memory_effects = #llvm.memory_effects<other = none, argMem = read,
// CHECK-SAME: passthrough = ["nounwind"]
// CHECK-SAME: personality = @__gxx_personality_v0
// CHECK-SAME: target_features = #llvm.target_features<["+fp-armv8", "+neon"]>
llvm.func weak_odr fastcc @envelope(%p: !llvm.ptr {llvm.noalias, llvm.readonly},
                                    %n: i32) -> (i32 {llvm.signext})
    comdat(@__llvm_comdat::@envelope)
    attributes {
      denormal_fpenv = #llvm.denormal_fpenv<default_output_mode = preservesign,
                                            default_input_mode = ieee,
                                            float_output_mode = ieee,
                                            float_input_mode = ieee>,
      fp_contract = "fast",
      memory_effects = #llvm.memory_effects<other = none, argMem = read,
                                            inaccessibleMem = none,
                                            errnoMem = none,
                                            targetMem0 = none,
                                            targetMem1 = none>,
      passthrough = ["nounwind"],
      personality = @__gxx_personality_v0,
      target_features = #llvm.target_features<["+fp-armv8", "+neon"]>
    } {
  %zero = llvm.mlir.constant(0 : i32) : i32
  %positive = llvm.icmp "sgt" %n, %zero : i32
  llvm.cond_br %positive, ^load, ^empty
^load:
  %value = llvm.load %p : !llvm.ptr -> i32
  llvm.br ^exit(%value : i32)
^empty:
  llvm.br ^exit(%zero : i32)
^exit(%result: i32):
  // CHECK: scf.if
  // CHECK-NOT: llvm.cond_br
  // CHECK: llvm.return
  llvm.return %result : i32
}

// An LLVM aggregate signature or a variadic tail is an ABI fact, not a
// statement about the body. Each callable is structured and normalized where
// it stands rather than being skipped for holding a signature another dialect
// could not mirror.

// CHECK-LABEL: llvm.func @aggregate_abi
// CHECK-SAME: !llvm.struct<(i32, i32)>
// CHECK-SAME: -> !llvm.struct<(i32, i32)>
// CHECK-NOT: llvm.cond_br
// CHECK: scf.if
// CHECK: arith.addi
// CHECK: llvm.return
llvm.func @aggregate_abi(%pair: !llvm.struct<(i32, i32)>, %flag: i1)
    -> !llvm.struct<(i32, i32)> {
  llvm.cond_br %flag, ^scale, ^exit(%pair : !llvm.struct<(i32, i32)>)
^scale:
  %low = llvm.extractvalue %pair[0] : !llvm.struct<(i32, i32)>
  %doubled = llvm.add %low, %low : i32
  %scaled = llvm.insertvalue %doubled, %pair[0] : !llvm.struct<(i32, i32)>
  llvm.br ^exit(%scaled : !llvm.struct<(i32, i32)>)
^exit(%result: !llvm.struct<(i32, i32)>):
  llvm.return %result : !llvm.struct<(i32, i32)>
}

// CHECK-LABEL: llvm.func @variadic_abi
// CHECK-SAME: i32, ...)
// CHECK-NOT: llvm.cond_br
// CHECK: scf.if
// CHECK: arith.addi
// CHECK: llvm.return
llvm.func @variadic_abi(%count: i32, ...) -> i32 {
  %zero = llvm.mlir.constant(0 : i32) : i32
  %positive = llvm.icmp "sgt" %count, %zero : i32
  llvm.cond_br %positive, ^bump, ^exit(%zero : i32)
^bump:
  %doubled = llvm.add %count, %count : i32
  llvm.br ^exit(%doubled : i32)
^exit(%result: i32):
  llvm.return %result : i32
}
