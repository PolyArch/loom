add_library(LoomDSEModelingGenerator
  ../../GroundTruthPlan.cpp
  ../../ModelParameterCalibrationAcquisition.cpp
  ModelParameterTrainingCandidateGenerator.cpp
)

target_include_directories(LoomDSEModelingGenerator PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomDSEModelingGenerator PUBLIC
  LoomCommon
  LoomConfig
  LoomDSE
  LoomDSECandidateGenerator
  LoomDSEPolicy
  LoomEvaluation
  LoomStructuredFabricAnalyticModel
  LLVMSupport
)
