add_mlir_library(LoomFrontendAnalysis
  ByteAddressDomain.cpp
  CountedLoopProjection.cpp
  DenseParallelMemoryProjection.cpp
  MemoryAddressProjection.cpp
  PointerLoopProjection.cpp
  MemoryProvenance.cpp
  StoredMemoryProvenance.cpp
  StructuredProtocolDependencies.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Frontend/Analysis
)

target_link_libraries(LoomFrontendAnalysis PUBLIC
  LoomCommon
  MLIRDataflow
  MLIRLoom
)
mlir_target_link_libraries(LoomFrontendAnalysis PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRFunctionInterfaces
  MLIRIR
  MLIRInferIntRangeInterface
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTargetLLVMIRExport
  MLIRSupport
)
