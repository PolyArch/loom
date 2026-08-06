add_library(LoomDSEFeedback
  SpatialMappingFeedbackCandidateGenerator.cpp
)

target_include_directories(LoomDSEFeedback PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomDSEFeedback PUBLIC
  LoomDSE
  LoomDSECompilerGenerator
  LoomFabricArtifact
  LoomFrontendCompilation
  LoomMappingArtifact
  LoomPnR
  MLIRDataflowTransforms
  LLVMSupport
)
