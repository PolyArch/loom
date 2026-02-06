// COMP_INSTANCE_ILLEGAL_TARGET is currently unreachable through normal MLIR
// text input. The verifier checks whether the resolved target is a
// fabric.add_tag, fabric.map_tag, or fabric.del_tag, but none of these ops
// carry the Symbol trait, so SymbolTable::lookupNearestSymbolFrom will never
// resolve them. The lookup returns nullptr, triggering COMP_INSTANCE_UNRESOLVED
// before reaching the ILLEGAL_TARGET check.
//
// This red test is intentionally disabled (no RUN line) because there is no
// valid MLIR text input that can trigger this error code.
//
// If the ODS definition is later changed to give tag ops the Symbol trait,
// the following would be the expected test:
//
//   // RUN: not loom --adg %s 2>&1 | FileCheck %s
//   // CHECK: COMP_INSTANCE_ILLEGAL_TARGET
//
//   fabric.add_tag @my_tag  // hypothetical: add_tag with Symbol trait
//   fabric.module @top(%v: i32) -> (!dataflow.tagged<i32, i4>) {
//     %out = fabric.instance @my_tag(%v) : (i32) -> (!dataflow.tagged<i32, i4>)
//     fabric.yield %out : !dataflow.tagged<i32, i4>
//   }
