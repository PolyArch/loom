add_mlir_library(LoomPnR
  Mapping.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/PnR

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRBuiltinToLLVMIRTranslation
  MLIRDataflow
  MLIRDLTIDialect
  MLIRFabric
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRParser
  MLIRSCFDialect
  MLIRSupport
  LLVMSupport
)
