add_library(LoomDSEHardwareGenerator
  FabricTemplateCandidateGenerator.cpp
  HardwareDecision.cpp
  SpatialMicroarchitectureCandidateGenerator.cpp
  SpatialTopologyCandidateGenerator.cpp
  SystemCompositionCandidateGenerator.cpp
)

target_include_directories(LoomDSEHardwareGenerator PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomDSEHardwareGenerator PUBLIC
  LoomADGBuilder
  LoomCommon
  LoomDSECandidateGenerator
  LoomFabricArtifact
  LLVMSupport
)
