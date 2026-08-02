set(LLVM_TARGET_DEFINITIONS MappingDialect.td)
mlir_tablegen(MappingDialect.h.inc -gen-dialect-decls -dialect=mapping)
mlir_tablegen(MappingDialect.cpp.inc -gen-dialect-defs -dialect=mapping)
add_public_tablegen_target(MLIRMappingDialectIncGen)

set(LLVM_TARGET_DEFINITIONS MappingEnums.td)
mlir_tablegen(MappingEnums.h.inc -gen-enum-decls)
mlir_tablegen(MappingEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRMappingEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS MappingAttrs.td)
mlir_tablegen(MappingAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=mapping)
mlir_tablegen(MappingAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=mapping)
add_public_tablegen_target(MLIRMappingAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS MappingOps.td)
mlir_tablegen(MappingOps.h.inc -gen-op-decls)
mlir_tablegen(MappingOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRMappingOpsIncGen)
