set(LLVM_TARGET_DEFINITIONS FabricDialect.td)
mlir_tablegen(FabricDialect.h.inc -gen-dialect-decls -dialect=fabric)
mlir_tablegen(FabricDialect.cpp.inc -gen-dialect-defs -dialect=fabric)
add_public_tablegen_target(FccFabricDialectIncGen)

set(LLVM_TARGET_DEFINITIONS FabricTypes.td)
mlir_tablegen(FabricTypes.h.inc -gen-typedef-decls -typedefs-dialect=fabric)
mlir_tablegen(FabricTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=fabric)
add_public_tablegen_target(FccFabricTypesIncGen)

set(LLVM_TARGET_DEFINITIONS FabricOps.td)
mlir_tablegen(FabricOps.h.inc -gen-op-decls)
mlir_tablegen(FabricOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(FccFabricOpsIncGen)
