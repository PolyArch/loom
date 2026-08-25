add_executable(loom-evaluation-case-test
  CaseTest.cpp
)

target_include_directories(loom-evaluation-case-test PRIVATE
  ${PROJECT_SOURCE_DIR}/lib/Common
)

target_link_libraries(loom-evaluation-case-test PRIVATE
  LoomConfig
  LoomEvaluation
  LoomImplementationPlatform
  LLVMSupport
)

add_executable(loom-evaluation-metric-test
  MetricTest.cpp
)

target_link_libraries(loom-evaluation-metric-test PRIVATE
  LoomEvaluation
  LLVMSupport
)

add_executable(loom-evaluation-evidence-test
  EvidenceTest.cpp
)

target_link_libraries(loom-evaluation-evidence-test PRIVATE
  LoomConfig
  LoomEvaluation
  LoomExternalTool
  LLVMSupport
)

add_executable(loom-evaluation-physical-rail-analysis-test
  PhysicalRailAnalysisTest.cpp
)

target_link_libraries(loom-evaluation-physical-rail-analysis-test PRIVATE
  LoomConfig
  LoomEvaluation
  LoomHardwareImplementation
  LoomImplementationPlatform
  LLVMSupport
)

add_executable(loom-evaluation-production-registry-test
  ProductionRegistryTest.cpp
)

target_link_libraries(loom-evaluation-production-registry-test PRIVATE
  LoomConfig
  LoomEvaluation
  LoomStructuredFabricAnalyticModel
  LLVMSupport
)

add_executable(loom-evaluation-fabric-fast-evaluation-test
  FabricFastEvaluationTest.cpp
)

target_link_libraries(loom-evaluation-fabric-fast-evaluation-test PRIVATE
  LoomADGBuilder
  LoomFabricArtifact
  LoomStructuredFabricAnalyticModel
  MLIRDataflow
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRLoom
  MLIRParser
  LLVMSupport
)
