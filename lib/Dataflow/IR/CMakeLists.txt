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
  DataflowChannelOps.cpp
  DataflowDialect.cpp
  DataflowGraphValidation.cpp
  DataflowProgramValidation.cpp
  DataflowInterfaces.cpp
  DataflowFunctionLikeOps.cpp
  DataflowOps.cpp
  DataflowThreadCompletion.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Dataflow

  DEPENDS
  MLIRDataflowDialectIncGen
  MLIRDataflowInterfacesIncGen
  MLIRDataflowOpsIncGen

  LINK_LIBS PUBLIC
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
  MLIRUBDialect
)
