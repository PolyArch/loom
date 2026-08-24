add_library(LoomDSEFeedback
  ../JointHardwareReopen.cpp
  ../JointHardwareReopenExecution.cpp
  ../JointMappingMigration.cpp
  SpatialOperandQueueRuntimeFeedback.cpp
  SpatialTransportRuntimeFeedback.cpp
  SpatialTransportRuntimeRepair.cpp
  SpatialMappingFeedbackCandidateGenerator.cpp
  TechMappingHardwareFeedback.cpp
)

target_include_directories(LoomDSEFeedback PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomDSEFeedback PUBLIC
  LoomDSE
  LoomDSECompilerGenerator
  LoomDSEHardwareGenerator
  LoomFabricArtifact
  LoomFrontendCompilation
  LoomMappingArtifact
  LoomTechMapping
  LoomPnR
  MLIRDataflowTransforms
  LLVMSupport
)
