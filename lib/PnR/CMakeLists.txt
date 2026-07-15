add_mlir_library(LoomPnR
  Mapping.cpp
  MappingIO.cpp
  MappingRouting.cpp
  MappingEstimateReport.cpp
  MappingEstimator.cpp

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
  MLIRMathDialect
  MLIRParser
  MLIRSCFDialect
  MLIRSupport
  MLIRUBDialect
  LLVMSupport
)
