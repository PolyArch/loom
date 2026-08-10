add_library(LoomRuntimePlatformBinding
  RuntimePlatformBinding.cpp
  RuntimeProvider.cpp
)

target_include_directories(LoomRuntimePlatformBinding PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomRuntimePlatformBinding PUBLIC
  LoomCommon
  LoomFabricArtifact
  LoomHardwareConfiguration
  LoomHardwareImplementation
  LLVMSupport
)
