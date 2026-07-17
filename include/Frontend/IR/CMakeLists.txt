set(LLVM_TARGET_DEFINITIONS LoomDialect.td)
mlir_tablegen(LoomDialect.h.inc -gen-dialect-decls -dialect=loom)
mlir_tablegen(LoomDialect.cpp.inc -gen-dialect-defs -dialect=loom)
add_public_tablegen_target(MLIRLoomDialectIncGen)

set(LLVM_TARGET_DEFINITIONS LoomOps.td)
mlir_tablegen(LoomOps.h.inc -gen-op-decls)
mlir_tablegen(LoomOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRLoomOpsIncGen)
