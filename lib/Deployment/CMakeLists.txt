add_library(LoomDeployment
  HardwareConfigurationImage.cpp
)

target_include_directories(LoomDeployment PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomDeployment PUBLIC
  LoomCommon
  LoomHardwareConfiguration
  LoomMappingArtifact
  LLVMSupport
)
