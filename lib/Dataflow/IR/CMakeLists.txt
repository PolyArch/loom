add_mlir_library(MLIRDataflowEnums
  DataflowEnums.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Dataflow

  DEPENDS
  MLIRDataflowEnumsIncGen

)

mlir_target_link_libraries(MLIRDataflowEnums PUBLIC
  MLIRIR
  MLIRSupport
)

add_mlir_dialect_library(MLIRDataflow
  DataflowActorSemantics.cpp
  DataflowActivityDefinedness.cpp
  DataflowHandshakeSemantics.cpp
  DataflowArtifactLocalReference.cpp
  DataflowCanonicalArtifact.cpp
  DataflowCanonicalBytecode.cpp
  DataflowCanonicalLabeling.cpp
  DataflowChannelOps.cpp
  DataflowDialect.cpp
  DataflowGraphCausality.cpp
  DataflowGraphValidation.cpp
  DataflowInvocationPathProjection.cpp
  DataflowLogicalDomainProjection.cpp
  DataflowInterfaces.cpp
  DataflowProgramValidation.cpp
  DataflowFunctionLikeOps.cpp
  DataflowMemoryContracts.cpp
  DataflowOps.cpp
  DataflowReferenceCodec.cpp
  DataflowServiceSchema.cpp
  DataflowStaticScheduleAnalysis.cpp
  DataflowStructuralView.cpp
  DataflowSyncRendezvous.cpp
  OperationSchema.cpp
  OperationSchemaAtomCodec.cpp
  OperationSchemaCodec.cpp
  OperationSchemaTypeCodec.cpp
  DataflowThreadCompletion.cpp
  DataflowVectorSemantics.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Dataflow

  DEPENDS
  MLIRDataflowAttrsIncGen
  MLIRDataflowDialectIncGen
  MLIRDataflowInterfacesIncGen
  MLIRDataflowOpsIncGen
  LoomOperationSchemasIncGen

)

target_link_libraries(MLIRDataflow PUBLIC
  LoomCommon
  MLIRDataflowEnums
)
mlir_target_link_libraries(MLIRDataflow PUBLIC
  MLIRRegisterAllDialects
  MLIRCallInterfaces
  MLIRArithDialect
  MLIRControlFlowInterfaces
  MLIRFunctionInterfaces
  MLIRIR
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRUBDialect
  MLIRVectorDialect
)
