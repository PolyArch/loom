add_library(LoomHardwareImplementation
  HardwareImplementation.cpp
)

target_include_directories(LoomHardwareImplementation PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomHardwareImplementation PUBLIC
  LoomCommon
  LoomFabricArtifact
  LoomHardwareConfiguration
  LoomImplementationPlatform
  LLVMSupport
)
