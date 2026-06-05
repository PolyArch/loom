add_llvm_executable(loom-pnr-map
  loom-pnr-map.cpp
)

llvm_update_compile_flags(loom-pnr-map)

target_link_libraries(loom-pnr-map
  PRIVATE
  LoomPnR
  MLIRArithDialect
  MLIRDataflow
  MLIRDLTIDialect
  MLIRFabric
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRParser
  MLIRSCFDialect
  MLIRSupport
  MLIRTransforms
)

mlir_check_all_link_libraries(loom-pnr-map)
