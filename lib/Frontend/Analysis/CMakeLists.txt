add_mlir_library(LoomFrontendAnalysis
  DenseParallelMemoryProjection.cpp
  MemoryProvenance.cpp
  StructuredProtocolDependencies.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Frontend/Analysis
)

target_link_libraries(LoomFrontendAnalysis PUBLIC
  MLIRDataflow
  MLIRLoom
)
mlir_target_link_libraries(LoomFrontendAnalysis PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRFunctionInterfaces
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRSupport
)
