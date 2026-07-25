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
  DataflowActorSemantics.cpp
  DataflowArtifactLocalReference.cpp
  DataflowChannelOps.cpp
  DataflowDialect.cpp
  DataflowGraphValidation.cpp
  DataflowProgramValidation.cpp
  DataflowInterfaces.cpp
  DataflowFunctionLikeOps.cpp
  DataflowMemoryContracts.cpp
  DataflowOps.cpp
  DataflowReferenceCodec.cpp
  DataflowServiceSchema.cpp
  DataflowThreadCompletion.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Dataflow

  DEPENDS
  MLIRDataflowAttrsIncGen
  MLIRDataflowDialectIncGen
  MLIRDataflowInterfacesIncGen
  MLIRDataflowOpsIncGen

  LINK_LIBS PUBLIC
  LoomCommon
  MLIRCallInterfaces
  MLIRArithDialect
  MLIRControlFlowInterfaces
  MLIRDataflowEnums
  MLIRFunctionInterfaces
  MLIRIR
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRSideEffectInterfaces
  MLIRSupport
)
