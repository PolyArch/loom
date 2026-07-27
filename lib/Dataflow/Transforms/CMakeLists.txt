add_mlir_library(MLIRDataflowTransforms
  DataflowRewritePass.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Dataflow

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRSupport
)

target_link_libraries(MLIRDataflowTransforms PUBLIC
  MLIRDataflow
)
