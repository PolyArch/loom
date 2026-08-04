add_library(LoomDSE
  StructuredEvaluationAcquisition.cpp
  StructuredOwnership.cpp
  StructuredOwnershipInvocation.cpp
)

target_include_directories(LoomDSE PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomDSE PUBLIC
  LoomCommon
  LoomConfig
  LoomDSECandidateGenerator
  LoomDSEObjective
  LoomDSEPolicy
  LoomEvaluation
  LoomFrontendCompilation
  LoomNativeSimulationOracle
  LoomSimulator
  LoomStructuredFabricAnalyticModel
  LLVMSupport
)

add_subdirectory(Generators/Mapping)
add_subdirectory(Generators/Compiler)
