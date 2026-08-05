add_mlir_library(MLIRDataflowTransforms
  DataflowRewritePass.cpp
  DataflowRewriteDecision.cpp
  DataflowVectorDecomposition.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Dataflow

)

target_link_libraries(MLIRDataflowTransforms PUBLIC
  MLIRDataflow
)
mlir_target_link_libraries(MLIRDataflowTransforms PUBLIC
  MLIRIR
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRVectorDialect
)
