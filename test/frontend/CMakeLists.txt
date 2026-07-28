add_subdirectory(payload)

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

add_executable(loom-structured-ownership-accounting-test
  StructuredOwnershipAccountingTest.cpp
)

target_link_libraries(loom-structured-ownership-accounting-test PRIVATE
  LoomADGBuilder
  LoomCommon
  LoomDSE
  LoomFrontendCompilation
  LoomStructuredFabricAnalyticModel
  LLVMCore
  LLVMIRReader
  LLVMSupport
)

add_executable(loom-structured-ownership-evaluation-test
  StructuredOwnershipEvaluationTest.cpp
)

target_link_libraries(loom-structured-ownership-evaluation-test PRIVATE
  LoomADGBuilder
  LoomCommon
  LoomDSE
  LoomFrontendCompilation
  LoomStructuredFabricAnalyticModel
  LLVMCore
  LLVMIRReader
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
