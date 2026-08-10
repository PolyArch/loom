add_library(LoomDeployment
  Deployment.cpp
  DeploymentCodec.cpp
  DeploymentPipeline.cpp
  DeploymentReference.cpp
  DeploymentSpatialLaunchSelection.cpp
  ExecutableLeaves.cpp
  HardwareConfigurationImage.cpp
  Package.cpp
  RuntimeImageDerivation.cpp
)

target_include_directories(LoomDeployment PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomDeployment PUBLIC
  LoomCommon
  LoomFrontendCompilation
  LoomFrontendExecutable
  LoomHardwareConfiguration
  LoomHardwareImplementation
  LoomImplementationPlatform
  LoomMappingArtifact
  LoomRuntimePlatformBinding
  MLIRDataflow
  MLIRLLVMDialect
  LLVM
)
