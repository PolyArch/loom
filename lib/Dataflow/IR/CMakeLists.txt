add_mlir_dialect_library(MLIRDataflow
  DataflowDialect.cpp
  DataflowEnums.cpp
  DataflowFunctionLikeOps.cpp
  DataflowOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Dataflow

  DEPENDS
  MLIRDataflowDialectIncGen
  MLIRDataflowOpsIncGen

  LINK_LIBS PUBLIC
  MLIRCallInterfaces
  MLIRFabric
  MLIRFunctionInterfaces
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRSupport
)
