add_mlir_library(LoomPnR
  FrozenComputeDomains.cpp
  FrozenRealizationGraph.cpp
  Mapping.cpp
  MappingHardware.cpp
  MappingIO.cpp
  MappingRouting.cpp
  MappingEstimateReport.cpp
  MappingEstimator.cpp
  PnrIndex.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/PnR

  LINK_LIBS PUBLIC
  LoomMapping
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

target_include_directories(LoomPnR PRIVATE
  ${LOOM_SOURCE_DIR}/lib
)
