add_executable(loom-dse-objective-test
  ObjectiveTest.cpp
)

add_executable(loom-dse-candidate-generator-test
  CandidateGeneratorTest.cpp
)

add_executable(loom-dse-production-candidate-generator-registration-test
  ProductionCandidateGeneratorRegistrationTest.cpp
)

add_executable(loom-dse-hardware-candidate-generator-test
  HardwareCandidateGeneratorTest.cpp
)

add_executable(loom-dse-portable-system-rtl-generator-test
  PortableSystemRtlGeneratorTest.cpp
)

add_executable(loom-dse-model-parameter-training-test
  ModelParameterTrainingTest.cpp
)

add_executable(loom-dse-plan-test
  PlanTest.cpp
)

add_executable(loom-dse-execution-journal-test
  ExecutionJournalTest.cpp
)

add_executable(loom-dse-plan-executor-test
  PlanExecutionTestSupport.cpp
  PlanExecutorTest.cpp
)

add_executable(loom-dse-site-scheduler-test
  SiteSchedulerTest.cpp
)

add_executable(loom-dse-campaign-runner-test
  CampaignRunnerTest.cpp
  PlanExecutionTestSupport.cpp
)

add_executable(loom-dse-invocation-manifest-test
  InvocationManifestTest.cpp
)

add_executable(loom-dse-evidence-obligation-test
  EvidenceObligationTest.cpp
)

add_executable(loom-dse-promotion-acquisition-execution-test
  PromotionAcquisitionExecutionTest.cpp
)

add_executable(loom-dse-resolved-config-view-test
  ResolvedConfigViewTest.cpp
)

add_executable(loom-dse-promotion-test
  PromotionTest.cpp
)

add_executable(loom-root-complete-tech-mapping-generator-test
  RootCompleteTechMappingGeneratorTest.cpp
)

add_library(LoomRootCompleteSpatialPnrTestSupport
  RootCompleteSpatialPnrTestSupport.cpp
)

target_include_directories(LoomRootCompleteSpatialPnrTestSupport PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(LoomRootCompleteSpatialPnrTestSupport PUBLIC
  LoomADGBuilder
  LoomFabricArtifact
  LLVMSupport
)

add_executable(loom-root-complete-spatial-pnr-generator-test
  RootCompleteSpatialFeedbackTestSupport.cpp
  RootCompleteSpatialPnrGeneratorTest.cpp
)

target_link_libraries(loom-dse-objective-test PRIVATE
  LoomDSEObjective
  LLVMSupport
)

target_link_libraries(loom-dse-candidate-generator-test PRIVATE
  LoomDSECandidateGenerator
  LoomExternalTool
  LLVMSupport
)

target_link_libraries(loom-dse-production-candidate-generator-registration-test PRIVATE
  LoomDSECompilerGenerator
  LoomDSEFeedback
  LoomDSEHardwareGenerator
  LoomDSEMappingGenerator
  LoomDSEModelingGenerator
  LLVMSupport
)

target_link_libraries(loom-dse-hardware-candidate-generator-test PRIVATE
  LoomADGBuilder
  LoomDSEHardwareGenerator
  LoomFabricArtifact
  LLVMSupport
)

target_link_libraries(loom-dse-portable-system-rtl-generator-test PRIVATE
  LoomMappedRtlSimulationTestSupport
  LoomDSEHardwareGenerator
  LLVMSupport
)

target_link_libraries(loom-dse-model-parameter-training-test PRIVATE
  LoomDSEModelingGenerator
  LLVMSupport
)

target_link_libraries(loom-dse-plan-test PRIVATE
  LoomDSE
  LLVMSupport
)

target_link_libraries(loom-dse-execution-journal-test PRIVATE
  LoomDSE
  LLVMSupport
)

target_link_libraries(loom-dse-plan-executor-test PRIVATE
  LoomDSE
  LLVMSupport
)

target_link_libraries(loom-dse-site-scheduler-test PRIVATE
  LoomDSE
  LLVMSupport
)

target_link_libraries(loom-dse-campaign-runner-test PRIVATE
  LoomDSE
  LLVMSupport
)

target_link_libraries(loom-dse-invocation-manifest-test PRIVATE
  LoomDSE
  LLVMSupport
)

target_link_libraries(loom-dse-evidence-obligation-test PRIVATE
  LoomDSE
  LLVMSupport
)

target_link_libraries(loom-dse-promotion-acquisition-execution-test PRIVATE
  LoomDSE
  LLVMSupport
)

target_link_libraries(loom-dse-resolved-config-view-test PRIVATE
  LoomDSE
  LLVMSupport
)

target_link_libraries(loom-dse-promotion-test PRIVATE
  LoomDSE
  LLVMSupport
)

target_link_libraries(loom-root-complete-tech-mapping-generator-test PRIVATE
  LoomADGBuilder
  LoomDSEMappingGenerator
  LoomMappingArtifact
  LoomTechMapping
  MLIRArithDialect
  MLIRDataflow
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  LLVMSupport
)

target_link_libraries(loom-root-complete-spatial-pnr-generator-test PRIVATE
  LoomRootCompleteSpatialPnrTestSupport
  LoomADGBuilder
  LoomDSE
  LoomDSEFeedback
  LoomDSEMappingGenerator
  LoomMappingArtifact
  LoomMappingInspection
  LoomPnRGenerator
  LoomStructuredFabricAnalyticModel
  LoomTechMapping
  MLIRArithDialect
  MLIRDataflow
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  LLVMSupport
)
