add_executable(loom-dataflow-memory-effect-test
  DataflowMemoryEffectTest.cpp
)

target_link_libraries(loom-dataflow-memory-effect-test PRIVATE
  MLIRDataflow
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRSupport
  LLVMSupport
)

add_executable(loom-dataflow-service-schema-test
  DataflowServiceSchemaTest.cpp
)

target_link_libraries(loom-dataflow-service-schema-test PRIVATE
  LoomCommon
  MLIRDLTIDialect
  MLIRDataflow
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
)

add_executable(loom-dataflow-canonical-reference-test
  CanonicalReferenceTest.cpp
)

target_link_libraries(loom-dataflow-canonical-reference-test PRIVATE
  LoomCommon
  MLIRDataflow
  MLIRIR
  MLIRSupport
  LLVMSupport
)

add_executable(loom-dataflow-canonical-artifact-test
  CanonicalArtifactTest.cpp
)

target_link_libraries(loom-dataflow-canonical-artifact-test PRIVATE
  LoomCommon
  MLIRArithDialect
  MLIRDataflow
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
)

add_executable(loom-dataflow-channel-create-test
  ChannelCreateTest.cpp
)

target_link_libraries(loom-dataflow-channel-create-test PRIVATE
  LoomCommon
  MLIRArithDialect
  MLIRDataflow
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
)

add_executable(loom-dataflow-activity-definedness-test
  ActivityDefinednessTest.cpp
)

add_executable(loom-dataflow-ordered-cardinality-handshake-test
  OrderedCardinalityHandshakeTest.cpp
)

add_executable(loom-dataflow-rewrite-decision-test
  DataflowRewriteDecisionTest.cpp
)

add_executable(loom-dataflow-sync-rewrite-test
  SyncRewriteTest.cpp
)

add_executable(loom-dataflow-cardinality-rewrite-test
  CardinalityRewriteTest.cpp
)

add_executable(loom-dataflow-fanout-rewrite-test
  FanoutRewriteTest.cpp
)

add_executable(loom-dataflow-graph-definition-rewrite-test
  GraphDefinitionRewriteTest.cpp
)

add_executable(loom-dataflow-vector-decomposition-rewrite-test
  VectorDecompositionRewriteTest.cpp
)

target_link_libraries(loom-dataflow-activity-definedness-test PRIVATE
  LoomCommon
  MLIRArithDialect
  MLIRDataflow
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRParser
  MLIRSupport
  MLIRUBDialect
  LLVMSupport
)

target_link_libraries(loom-dataflow-ordered-cardinality-handshake-test PRIVATE
  MLIRDataflow
  LLVMSupport
)

target_link_libraries(loom-dataflow-rewrite-decision-test PRIVATE
  LoomCommon
  MLIRDataflow
  MLIRDataflowTransforms
  LLVMSupport
)

target_link_libraries(loom-dataflow-sync-rewrite-test PRIVATE
  LoomCommon
  LoomSimulator
  MLIRDataflow
  MLIRDataflowTransforms
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
)

target_link_libraries(loom-dataflow-cardinality-rewrite-test PRIVATE
  LoomCommon
  LoomSimulator
  MLIRArithDialect
  MLIRDataflow
  MLIRDataflowTransforms
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
)

target_link_libraries(loom-dataflow-fanout-rewrite-test PRIVATE
  LoomCommon
  LoomSimulator
  MLIRArithDialect
  MLIRDataflow
  MLIRDataflowTransforms
  MLIRIR
  MLIRLLVMDialect
  MLIRParser
  MLIRSupport
  LLVMSupport
)

target_link_libraries(loom-dataflow-graph-definition-rewrite-test PRIVATE
  LoomCommon
  LoomSimulator
  MLIRArithDialect
  MLIRDataflow
  MLIRDataflowTransforms
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
)

target_link_libraries(loom-dataflow-vector-decomposition-rewrite-test PRIVATE
  LoomCommon
  LoomSimulator
  MLIRArithDialect
  MLIRDataflow
  MLIRDataflowTransforms
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
)

add_executable(loom-dataflow-operation-schema-test
  OperationSchemaTest.cpp
)

target_link_libraries(loom-dataflow-operation-schema-test PRIVATE
  MLIRDataflow
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRUBDialect
  MLIRVectorDialect
  MLIRSupport
  LLVMSupport
)

add_executable(loom-dataflow-operation-schema-codec-test
  OperationSchemaCodecTest.cpp
)

target_link_libraries(loom-dataflow-operation-schema-codec-test PRIVATE
  LoomCommon
  MLIRDataflow
  MLIRArithDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRSupport
  LLVMSupport
)
