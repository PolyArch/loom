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
  LoomFrontendCompilation
  LoomNativeSimulationOracle
  LoomSimulator
  LoomStructuredFabricAnalyticModel
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
