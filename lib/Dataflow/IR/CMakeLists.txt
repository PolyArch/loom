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
  DataflowCanonicalArtifact.cpp
  DataflowCanonicalBytecode.cpp
  DataflowCanonicalLabeling.cpp
  DataflowChannelOps.cpp
  DataflowDialect.cpp
  DataflowGraphValidation.cpp
  DataflowInterfaces.cpp
  DataflowProgramValidation.cpp
  DataflowFunctionLikeOps.cpp
  DataflowMemoryContracts.cpp
  DataflowOps.cpp
  DataflowReferenceCodec.cpp
  DataflowServiceSchema.cpp
  DataflowStructuralView.cpp
  OperationSchema.cpp
  OperationSchemaAtomCodec.cpp
  OperationSchemaCodec.cpp
  DataflowThreadCompletion.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Dataflow

  DEPENDS
  MLIRDataflowAttrsIncGen
  MLIRDataflowDialectIncGen
  MLIRDataflowInterfacesIncGen
  MLIRDataflowOpsIncGen
  LoomOperationSchemasIncGen

  LINK_LIBS PUBLIC
  LoomCommon
  MLIRRegisterAllDialects
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
  MLIRVectorDialect
)
