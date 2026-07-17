add_mlir_library(MLIRDataflowEnums
  DataflowEnums.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Dataflow

  DEPENDS
  MLIRDataflowEnumsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)

add_mlir_dialect_library(MLIRDataflow
  DataflowDialect.cpp
  DataflowFunctionLikeOps.cpp
  DataflowOps.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Dataflow

  DEPENDS
  MLIRDataflowDialectIncGen
  MLIRDataflowOpsIncGen

  LINK_LIBS PUBLIC
  MLIRCallInterfaces
  MLIRDataflowEnums
  MLIRFabric
  MLIRFunctionInterfaces
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRSupport
)
