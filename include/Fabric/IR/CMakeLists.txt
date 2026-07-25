set(LLVM_TARGET_DEFINITIONS FabricDialect.td)
mlir_tablegen(FabricDialect.h.inc -gen-dialect-decls -dialect=fabric)
mlir_tablegen(FabricDialect.cpp.inc -gen-dialect-defs -dialect=fabric)
add_public_tablegen_target(MLIRFabricDialectIncGen)

set(LLVM_TARGET_DEFINITIONS FabricTypes.td)
mlir_tablegen(FabricTypes.h.inc -gen-typedef-decls -typedefs-dialect=fabric)
mlir_tablegen(FabricTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=fabric)
add_public_tablegen_target(MLIRFabricTypesIncGen)

set(LLVM_TARGET_DEFINITIONS
  ${LOOM_SOURCE_DIR}/include/Dataflow/IR/OperationSchemas.td)
tablegen(LOOM ImplementationFamilyEnum.td.inc
  -gen-implementation-family-enum)
add_public_tablegen_target(LoomImplementationFamilyEnumTdIncGen)

set(IMPLEMENTATION_FAMILY_ENUM_TD
  ${CMAKE_CURRENT_BINARY_DIR}/ImplementationFamilyEnum.td.inc)
set(LLVM_TARGET_DEFINITIONS FabricEnums.td)
mlir_tablegen(FabricEnums.h.inc -gen-enum-decls
  DEPENDS ${IMPLEMENTATION_FAMILY_ENUM_TD})
mlir_tablegen(FabricEnums.cpp.inc -gen-enum-defs
  DEPENDS ${IMPLEMENTATION_FAMILY_ENUM_TD})
add_public_tablegen_target(MLIRFabricEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS FabricAttrs.td)
mlir_tablegen(FabricAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=fabric
  DEPENDS ${IMPLEMENTATION_FAMILY_ENUM_TD})
mlir_tablegen(FabricAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=fabric
  DEPENDS ${IMPLEMENTATION_FAMILY_ENUM_TD})
add_public_tablegen_target(MLIRFabricAttrsIncGen)
add_dependencies(MLIRFabricAttrsIncGen MLIRFabricEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS FabricOps.td)
mlir_tablegen(FabricOps.h.inc -gen-op-decls)
mlir_tablegen(FabricOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRFabricOpsIncGen)
add_dependencies(MLIRFabricOpsIncGen MLIRFabricAttrsIncGen)

# The implementation-family registry is generated from the one canonical
# operation source; Fabric adds no second declaration of its own.
set(LLVM_TARGET_DEFINITIONS
  ${LOOM_SOURCE_DIR}/include/Dataflow/IR/OperationSchemas.td)
tablegen(LOOM ImplementationFamilies.inc -gen-implementation-families)
add_public_tablegen_target(LoomImplementationFamiliesIncGen)
