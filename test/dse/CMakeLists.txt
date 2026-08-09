add_executable(loom-dse-objective-test
  ObjectiveTest.cpp
)

add_executable(loom-dse-candidate-generator-test
  CandidateGeneratorTest.cpp
)

add_executable(loom-dse-production-candidate-generator-registration-test
  ProductionCandidateGeneratorRegistrationTest.cpp
)

add_executable(loom-dse-plan-test
  PlanTest.cpp
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

add_executable(loom-root-complete-spatial-pnr-generator-test
  RootCompleteSpatialFeedbackTestSupport.cpp
  RootCompleteSpatialPnrGeneratorTest.cpp
  RootCompleteSpatialPnrTestSupport.cpp
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
  LoomDSEMappingGenerator
  LLVMSupport
)

target_link_libraries(loom-dse-plan-test PRIVATE
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
