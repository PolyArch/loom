add_library(LoomDSEFeedback
  ../HardwareMutationRepairRecord.cpp
  ../JointHardwareReopen.cpp
  ../JointHardwareReopenCandidate.cpp
  ../JointHardwareReopenExecution.cpp
  ../JointHardwareRuntimeRepair.cpp
  ../JointMappingMigration.cpp
  ../JointRepairQuality.cpp
  ../ResourceTimeAdjacentMappingSelection.cpp
  SpatialFifoRuntimeFeedback.cpp
  SpatialOperandQueueRuntimeFeedback.cpp
  SpatialTransportRuntimeFeedback.cpp
  SpatialTransportRuntimeRepair.cpp
  SpatialTransportCegar.cpp
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
  LoomPnRCpSat
  MLIRDataflowTransforms
  LLVMSupport
)
