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

add_library(LoomRuntimeDeploymentLoader
  DeploymentLoader.cpp
  Providers/InProcessPlatform.cpp
)

target_include_directories(LoomRuntimeDeploymentLoader PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomRuntimeDeploymentLoader PUBLIC
  LoomCommon
  LoomDeployment
  LoomFabricArtifact
  LoomFrontendExecutable
  LoomHardwareConfiguration
  LoomHardwareImplementation
  LoomHardwareRTL
  LoomMappingArtifact
  LoomRuntimePlatformBinding
  LLVMSupport
)
