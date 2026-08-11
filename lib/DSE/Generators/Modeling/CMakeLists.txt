add_library(LoomDSEModelingGenerator
  ModelParameterTrainingCandidateGenerator.cpp
)

target_include_directories(LoomDSEModelingGenerator PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomDSEModelingGenerator PUBLIC
  LoomCommon
  LoomDSECandidateGenerator
  LoomEvaluation
  LoomStructuredFabricAnalyticModel
  LLVMSupport
)
