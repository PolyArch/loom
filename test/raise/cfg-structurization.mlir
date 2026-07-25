// RUN: split-file %s %t
// RUN: %loom-raise %t/counted.ll | FileCheck %s --check-prefix=LOOP
// RUN: %loom-raise %t/spin.ll | FileCheck %s --check-prefix=SPIN
// RUN: %loom-raise %t/irreducible.ll | FileCheck %s --check-prefix=UNDEF --implicit-check-not=ub.poison
// RUN: loom-raise-opt --loom-llvm-cf-to-cf --loom-lift-cf-to-scf %t/switch-carrier.mlir | FileCheck %s --check-prefix=SWITCH
// RUN: loom-raise-opt --loom-lift-cf-to-scf %t/preserved.mlir | FileCheck %s --check-prefix=PRESERVE
// RUN: loom-raise-opt --loom-lift-cf-to-scf %t/nested.mlir | FileCheck %s --check-prefix=NESTED --implicit-check-not=cf.cond_br

// Mechanical CFG recovery runs on the callable region where it stands. What
// an imported LLVM callable spells differently is respelled by an adapter, and
// a region whose structuring Loom cannot prove exact keeps its original
// control instead of failing the module.

// A counted cycle becomes scf.for, and the imported loop annotation moves from
// the latch branch that carried it to the structured loop that owns the cycle.
// LOOP: #[[ANNOTATION:.*]] = #llvm.loop_annotation<mustProgress = true>
// LOOP-LABEL: llvm.func @counted
// LOOP: scf.for
// LOOP: } {llvm.loop_annotation = #[[ANNOTATION]]}

// A statically infinite loop leaves the continuation of the structured loop
// unreachable. An imported callable states that with llvm.unreachable instead
// of returning a value its signature never produces.
// SPIN-LABEL: llvm.func @spin
// SPIN: scf.while
// SPIN: llvm.unreachable
// SPIN-NOT: llvm.return

// Structuring an irreducible cycle creates paths on which a value is never
// defined. Inside an imported callable that value is LLVM's own undef; poison
// would deepen it into deferred undefined behavior the source never stated.
// UNDEF-LABEL: llvm.func @irreducible
// UNDEF: llvm.mlir.undef : i32
// UNDEF: scf.while

// The structured switch reads its selector through index and its cases through
// 64-bit storage, so a selector wider than the target's index would silently
// drop high bits and keeps its exact cf form. An ordinary selector structures.
// SWITCH-LABEL: llvm.func @wide_selector
// SWITCH: cf.switch %arg0 : i128
// SWITCH-NOT: scf.index_switch
// SWITCH-LABEL: llvm.func @ordinary_selector
// SWITCH: scf.index_switch

// Preservation is per region, and a preserved region costs the module nothing:
// the structurable callable beside it is still recovered.
//
// A one-target llvm.indirectbr is side-effect free and implements the branch
// interface, so the transformation would splice its target away as an
// unconditional branch and erase the address semantics.
//
// A callable holding a value type LLVM cannot spell has no undef for it, so it
// keeps its exact original control rather than being given a stronger
// placeholder.
// PRESERVE-LABEL: llvm.func @structured_first
// PRESERVE: scf.if
// PRESERVE-LABEL: llvm.func @indirect_later
// PRESERVE: llvm.indirectbr
// PRESERVE-LABEL: llvm.func @index_valued
// PRESERVE: cf.cond_br
// PRESERVE-NOT: scf.
// PRESERVE-NOT: ub.poison

// Both the enclosing func.func and the nested imported llvm.func are multiblock
// and structurable, and the nested callable owns an annotated loop. Descendant-
// first publication structures the nested llvm.func and publishes it before the
// enclosing func.func is cloned, so the enclosing clone is taken from an
// original that already holds the structured descendant and publishing the
// ancestor cannot overwrite it. Keeping annotations per region is equally
// necessary: publishing the descendant invalidates its original blocks, which
// an ancestor plan must never inspect.
// NESTED-DAG: #[[INNER_ANNOTATION:.*]] = #llvm.loop_annotation<mustProgress = true>
// NESTED-LABEL: func.func @outer
// NESTED: scf.if
// NESTED-LABEL: llvm.func @inner
// NESTED: scf.while
// NESTED: } attributes {llvm.loop_annotation = #[[INNER_ANNOTATION]]}

//--- counted.ll
define void @counted(ptr %p) {
entry:
  br label %body

body:
  %i = phi i64 [ 0, %entry ], [ %next, %body ]
  %slot = getelementptr i32, ptr %p, i64 %i
  store i32 0, ptr %slot, align 4
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 8
  br i1 %done, label %exit, label %body, !llvm.loop !0

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.mustprogress"}

//--- spin.ll
define i32 @spin(ptr %flag) {
entry:
  br label %loop

loop:
  %observed = load volatile i32, ptr %flag, align 4
  br label %loop
}

//--- irreducible.ll
define i32 @irreducible(i1 %enter_right, i32 %seed) {
entry:
  br i1 %enter_right, label %right, label %left

left:
  %l = phi i32 [ %seed, %entry ], [ %rn, %right ]
  %ldone = icmp sgt i32 %l, 100
  br i1 %ldone, label %exit, label %right

right:
  %r = phi i32 [ %seed, %entry ], [ %l, %left ]
  %rn = add i32 %r, 1
  %rdone = icmp sgt i32 %rn, 200
  br i1 %rdone, label %exit, label %left

exit:
  %res = phi i32 [ %l, %left ], [ %rn, %right ]
  ret i32 %res
}

//--- switch-carrier.mlir
llvm.func @wide_selector(%v: i128) -> i32 {
  %a = llvm.mlir.constant(1 : i32) : i32
  %b = llvm.mlir.constant(2 : i32) : i32
  llvm.switch %v : i128, ^default [
    0: ^case
  ]
^case:
  llvm.return %a : i32
^default:
  llvm.return %b : i32
}

llvm.func @ordinary_selector(%v: i32) -> i32 {
  %a = llvm.mlir.constant(1 : i32) : i32
  %b = llvm.mlir.constant(2 : i32) : i32
  llvm.switch %v : i32, ^default [
    0: ^case
  ]
^case:
  llvm.return %a : i32
^default:
  llvm.return %b : i32
}

//--- preserved.mlir
llvm.func @structured_first(%c: i1) -> i32 {
  %z = llvm.mlir.constant(0 : i32) : i32
  cf.cond_br %c, ^yes, ^no
^yes:
  llvm.return %z : i32
^no:
  llvm.return %z : i32
}

llvm.func @indirect_later(%addr: !llvm.ptr) -> i32 {
  %z = llvm.mlir.constant(0 : i32) : i32
  llvm.indirectbr %addr : !llvm.ptr, [^target]
^target:
  llvm.return %z : i32
}

llvm.func @index_valued(%enter: i1) -> i64 {
  %one = arith.constant 1 : index
  cf.cond_br %enter, ^right(%one : index), ^left(%one : index)
^left(%l: index):
  %ldone = arith.cmpi sgt, %l, %one : index
  cf.cond_br %ldone, ^exit(%l : index), ^right(%l : index)
^right(%r: index):
  %rn = arith.addi %r, %one : index
  %rdone = arith.cmpi sgt, %rn, %one : index
  cf.cond_br %rdone, ^exit(%rn : index), ^left(%rn : index)
^exit(%res: index):
  %out = arith.index_cast %res : index to i64
  llvm.return %out : i64
}

//--- nested.mlir
#inner_annotation = #llvm.loop_annotation<mustProgress = true>

func.func @outer(%c: i1) -> i32 {
  %z = arith.constant 0 : i32
  cf.cond_br %c, ^a, ^b
^a:
  %one = arith.constant 1 : i32
  cf.br ^exit(%one : i32)
^b:
  cf.br ^exit(%z : i32)
^exit(%r: i32):
  builtin.module {
    llvm.func @inner(%limit: i32) -> i32 {
      %zero = arith.constant 0 : i32
      %inner_one = arith.constant 1 : i32
      cf.br ^loop(%zero : i32)
    ^loop(%iv: i32):
      %next = arith.addi %iv, %inner_one : i32
      %done = arith.cmpi eq, %next, %limit : i32
      cf.cond_br %done, ^iexit, ^loop(%next : i32) {
        llvm.loop_annotation = #inner_annotation
      }
    ^iexit:
      llvm.return %next : i32
    }
  }
  return %r : i32
}
