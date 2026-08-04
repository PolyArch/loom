add_library(LoomDSEObjective
  Objective.cpp
)

target_include_directories(LoomDSEObjective PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomDSEObjective PUBLIC
  LoomCommon
  LLVMSupport
)

add_library(LoomDSECandidateGenerator
  CandidateGenerator.cpp
)

target_include_directories(LoomDSECandidateGenerator PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomDSECandidateGenerator PUBLIC
  LoomCommon
  LLVMSupport
)

add_library(LoomDSE
  PreMappingExploration.cpp
  Promotion.cpp
  StructuredOwnership.cpp
)

target_include_directories(LoomDSE PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomDSE PUBLIC
  LoomCommon
  LoomDSEObjective
  LoomEvaluation
  LoomFrontendCompilation
  LoomNativeSimulationOracle
  LoomSimulator
  LoomStructuredFabricAnalyticModel
  LLVMSupport
)

add_subdirectory(Generators/Mapping)
