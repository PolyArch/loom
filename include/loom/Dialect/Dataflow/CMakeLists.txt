set(LLVM_TARGET_DEFINITIONS DataflowDialect.td)
mlir_tablegen(DataflowDialect.h.inc -gen-dialect-decls -dialect=dataflow)
mlir_tablegen(DataflowDialect.cpp.inc -gen-dialect-defs -dialect=dataflow)
add_public_tablegen_target(LoomDataflowDialectIncGen)

set(LLVM_TARGET_DEFINITIONS DataflowOps.td)
mlir_tablegen(DataflowOps.h.inc -gen-op-decls)
mlir_tablegen(DataflowOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(LoomDataflowOpsIncGen)
