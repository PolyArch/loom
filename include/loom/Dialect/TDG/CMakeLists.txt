set(LLVM_TARGET_DEFINITIONS TDGDialect.td)
mlir_tablegen(TDGDialect.h.inc -gen-dialect-decls -dialect=tdg)
mlir_tablegen(TDGDialect.cpp.inc -gen-dialect-defs -dialect=tdg)
add_public_tablegen_target(LoomTDGDialectIncGen)

set(LLVM_TARGET_DEFINITIONS TDGTypes.td)
mlir_tablegen(TDGTypes.h.inc -gen-typedef-decls -typedefs-dialect=tdg)
mlir_tablegen(TDGTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=tdg)
add_public_tablegen_target(LoomTDGTypesIncGen)

set(LLVM_TARGET_DEFINITIONS TDGOps.td)
mlir_tablegen(TDGOps.h.inc -gen-op-decls)
mlir_tablegen(TDGOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(LoomTDGOpsIncGen)
