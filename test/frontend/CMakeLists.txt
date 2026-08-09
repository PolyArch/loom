add_subdirectory(payload)

add_executable(loom-compiler-target-binding-test
  CompilerTargetBindingTest.cpp
)

target_link_libraries(loom-compiler-target-binding-test PRIVATE
  LoomADGBuilder
  LoomCommon
  LoomFrontendExecutable
)

add_executable(loom-instruction-core-binary-test
  InstructionCoreBinaryTest.cpp
)

target_link_libraries(loom-instruction-core-binary-test PRIVATE
  LoomADGBuilder
  LoomCommon
  LoomFrontendExecutable
  MLIRArithDialect
  MLIRDataflow
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRParser
  MLIRSupport
)

target_compile_definitions(loom-instruction-core-binary-test PRIVATE
  LOOM_TEST_LLD_PATH="${LLVM_TOOLS_BINARY_DIR}/ld.lld"
)

add_executable(loom-structured-program-artifact-test
  StructuredProgramArtifactTest.cpp
)

target_link_libraries(loom-structured-program-artifact-test PRIVATE
  LoomCommon
  LoomFrontendLowering
  MLIRLoom
  MLIRArithDialect
  MLIRDataflow
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRParser
  MLIRSupport
  LLVMSupport
)

add_executable(loom-pre-mapping-compilation-test
  PreMappingCompilationTest.cpp
)

target_link_libraries(loom-pre-mapping-compilation-test PRIVATE
  LoomADGBuilder
  LoomCommon
  LoomFrontendCompilation
  LLVMCore
  LLVMIRReader
  LLVMSupport
)

add_executable(loom-pointer-service-boundary-test
  PointerServiceBoundaryTest.cpp
)

target_link_libraries(loom-pointer-service-boundary-test PRIVATE
  LoomADGBuilder
  LoomCommon
  LoomFrontendCompilation
  LoomNativeSimulationOracle
  LoomSimulator
)

add_executable(loom-structured-address-index-narrowing-test
  StructuredAddressIndexNarrowingTest.cpp
)

target_include_directories(loom-structured-address-index-narrowing-test PRIVATE
  ${LOOM_SOURCE_DIR}/lib/Frontend/Compilation
)

target_link_libraries(loom-structured-address-index-narrowing-test PRIVATE
  LoomFrontendCompilation
  MLIRParser
)

add_executable(loom-structured-thread-domain-test
  StructuredThreadDomainTest.cpp
)

target_link_libraries(loom-structured-thread-domain-test PRIVATE
  LoomADGBuilder
  LoomCommon
  LoomFrontendCompilation
  LoomNativeSimulationOracle
  LoomSimulator
  MLIRParser
)

add_executable(loom-constant-callback-specialization-test
  ConstantCallbackSpecializationTest.cpp
)

target_link_libraries(loom-constant-callback-specialization-test PRIVATE
  LoomFrontendCompilation
  LLVMCore
  LLVMIRReader
  LLVMSupport
)

add_executable(loom-structured-ownership-accounting-test
  StructuredOwnershipAccountingTest.cpp
)

target_link_libraries(loom-structured-ownership-accounting-test PRIVATE
  LoomADGBuilder
  LoomCommon
  LoomDSE
  LoomDSECompilerGenerator
  LoomFrontendCompilation
  LoomNativeSimulationOracle
  LoomSimulator
  LoomStructuredFabricAnalyticModel
)

add_executable(loom-structured-ownership-evaluation-test
  StructuredOwnershipEvaluationTest.cpp
)

target_link_libraries(loom-structured-ownership-evaluation-test PRIVATE
  LoomADGBuilder
  LoomCommon
  LoomDSE
  LoomDSECompilerGenerator
  LoomFrontendCompilation
  LoomNativeSimulationOracle
  LoomSimulator
  LoomStructuredFabricAnalyticModel
  MLIRArithDialect
  MLIRParser
)

add_executable(loom-structured-ownership-lineage-index-test
  StructuredOwnershipLineageIndexTest.cpp
)

target_link_libraries(loom-structured-ownership-lineage-index-test PRIVATE
  LoomDSE
  MLIRDataflow
  MLIRDataflowTransforms
  LLVMSupport
)

add_executable(loom-structured-call-ownership-test
  StructuredCallOwnershipTest.cpp
)

target_link_libraries(loom-structured-call-ownership-test PRIVATE
  LoomADGBuilder
  LoomCommon
  LoomFrontendCompilation
  MLIRArithDialect
  MLIRDataflow
  MLIRIR
  MLIRLLVMDialect
  MLIRLoom
  MLIRParser
  MLIRSupport
  LLVMSupport
)

add_executable(loom-structured-schedule-generator-test
  StructuredScheduleGeneratorTest.cpp
)

target_link_libraries(loom-structured-schedule-generator-test PRIVATE
  LoomADGBuilder
  LoomCommon
  LoomDSECompilerGenerator
  LoomFrontendCompilation
  MLIRArithDialect
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRParser
  MLIRSCFDialect
  MLIRSupport
  LLVMSupport
)

add_executable(loom-structured-memory-communication-generator-test
  StructuredMemoryCommunicationGeneratorTest.cpp
)

target_link_libraries(loom-structured-memory-communication-generator-test PRIVATE
  LoomADGBuilder
  LoomCommon
  LoomDSECompilerGenerator
  LoomFrontendCompilation
  MLIRArithDialect
  MLIRDLTIDialect
  MLIRDataflow
  MLIRFuncDialect
  MLIRIR
  MLIRLoom
  MLIRMemRefDialect
  MLIRParser
  MLIRSupport
  LLVMSupport
)

add_executable(loom-structured-memory-communication-lineage-test
  StructuredMemoryCommunicationLineageTest.cpp
)

target_link_libraries(loom-structured-memory-communication-lineage-test PRIVATE
  LoomADGBuilder
  LoomCommon
  LoomDSECompilerGenerator
  LoomFrontendCompilation
  MLIRArithDialect
  MLIRDLTIDialect
  MLIRDataflow
  MLIRFuncDialect
  MLIRIR
  MLIRLoom
  MLIRMemRefDialect
  MLIRParser
  MLIRSCFDialect
  MLIRSupport
  LLVMSupport
)

add_executable(loom-structured-memory-layout-test
  StructuredMemoryLayoutTest.cpp
)

target_link_libraries(loom-structured-memory-layout-test PRIVATE
  LoomFrontendCompilation
  LoomNativeSimulationOracle
  LoomSimulator
  MLIRArithDialect
  MLIRDLTIDialect
  MLIRDataflow
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRLoom
  MLIRMemRefDialect
  MLIRParser
  MLIRSCFDialect
  MLIRSupport
  LLVMSupport
)

add_executable(loom-structured-memory-pipeline-test
  StructuredMemoryPipelineTest.cpp
)

target_link_libraries(loom-structured-memory-pipeline-test PRIVATE
  LoomFrontendCompilation
  LoomNativeSimulationOracle
  LoomSimulator
  MLIRArithDialect
  MLIRDLTIDialect
  MLIRDataflow
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRLoom
  MLIRMemRefDialect
  MLIRParser
  MLIRSCFDialect
  MLIRSupport
  LLVMSupport
)

add_executable(loom-structured-memory-channel-test
  StructuredMemoryChannelTest.cpp
)

target_link_libraries(loom-structured-memory-channel-test PRIVATE
  LoomFrontendCompilation
  LoomSimulator
  MLIRArithDialect
  MLIRDLTIDialect
  MLIRDataflow
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRLoom
  MLIRMemRefDialect
  MLIRParser
  MLIRSCFDialect
  MLIRSupport
  LLVMSupport
)

add_executable(loom-source-backed-attention-channel-test
  SourceBackedAttentionChannelTest.cpp
)

target_link_libraries(loom-source-backed-attention-channel-test PRIVATE
  LoomFrontendCompilation
  LoomSimulator
  MLIRDataflow
  MLIRIR
  MLIRLLVMDialect
  MLIRLoom
  MLIRParser
  MLIRSupport
  LLVMSupport
)

add_executable(loom-structured-execution-shape-generator-test
  StructuredExecutionShapeGeneratorTest.cpp
)

target_link_libraries(loom-structured-execution-shape-generator-test PRIVATE
  LoomADGBuilder
  LoomCommon
  LoomDSECompilerGenerator
  LoomFrontendCompilation
  LoomNativeSimulationOracle
  LoomSimulator
  MLIRArithDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRParser
  MLIRSupport
  LLVMSupport
)

add_executable(loom-structured-special-math-accuracy-generator-test
  StructuredSpecialMathAccuracyGeneratorTest.cpp
)

target_link_libraries(loom-structured-special-math-accuracy-generator-test PRIVATE
  LoomADGBuilder
  LoomCommon
  LoomDSECompilerGenerator
  LoomFrontendCompilation
  MLIRArithDialect
  MLIRDataflow
  MLIRIR
  MLIRLLVMDialect
  MLIRLoom
  MLIRMathDialect
  MLIRParser
  MLIRSupport
  LLVMSupport
)

add_executable(loom-dataflow-rewrite-generator-test
  DataflowRewriteGeneratorTest.cpp
)

target_link_libraries(loom-dataflow-rewrite-generator-test PRIVATE
  LoomADGBuilder
  LoomCommon
  LoomDSECompilerGenerator
  MLIRDataflow
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
)

add_executable(loom-frontend-dfg-integration-test
  FrontendDfgIntegrationTest.cpp
)

target_link_libraries(loom-frontend-dfg-integration-test PRIVATE
  LoomADGBuilder
  LoomCommon
  LoomFrontendCompilation
  LoomNativeSimulationOracle
  LoomSimulator
)

add_executable(loom-static-global-memory-test
  StaticGlobalMemoryTest.cpp
)

target_link_libraries(loom-static-global-memory-test PRIVATE
  LoomFrontendCompilation
  LLVMCore
  LLVMIRReader
  LLVMSupport
)
