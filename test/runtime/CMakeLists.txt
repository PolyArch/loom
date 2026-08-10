add_executable(loom-runtime-platform-binding-test
  RuntimePlatformBindingTest.cpp
)

target_link_libraries(loom-runtime-platform-binding-test PRIVATE
  LoomHardwareConfigurationTestSupport
  LoomRuntimePlatformBinding
  LoomHardwareImplementation
  MLIRDataflow
  MLIRFabric
  MLIRIR
  MLIRParser
  LLVMSupport
)

add_executable(loom-runtime-deployment-loader-test
  DeploymentLoaderTest.cpp
)

target_link_libraries(loom-runtime-deployment-loader-test PRIVATE
  LoomDeploymentTestSupport
  LoomRuntimeDeploymentLoader
)
