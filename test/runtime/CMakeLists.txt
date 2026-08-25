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

add_executable(loom-runtime-gem5-bridge-test
  Gem5BridgeTest.cpp
)

target_link_libraries(loom-runtime-gem5-bridge-test PRIVATE
  LoomRuntimePlatformBinding
  LLVMSupport
)

add_executable(loom-runtime-gem5-spatial-channel-test
  Gem5SpatialChannelTest.cpp
)

target_link_libraries(loom-runtime-gem5-spatial-channel-test PRIVATE
  LoomRuntimePlatformBinding
  LLVMSupport
)

add_executable(loom-runtime-ordered-channel-abi-test
  OrderedChannelABITest.cpp
)
target_link_libraries(loom-runtime-ordered-channel-abi-test PRIVATE
  LoomRuntimePlatformBinding
  LLVMSupport
)

add_executable(loom-runtime-gem5-simulation-binding-test
  Gem5SimulationBindingTest.cpp
)

target_link_libraries(loom-runtime-gem5-simulation-binding-test PRIVATE
  LoomHardwareConfigurationTestSupport
  LoomRuntimePlatformBinding
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

add_executable(loom-runtime-dynamic-work-test
  DynamicWorkExecutionTest.cpp
)

target_include_directories(loom-runtime-dynamic-work-test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../dse
)

target_link_libraries(loom-runtime-dynamic-work-test PRIVATE
  LoomDeploymentTestSupport
  LoomDSE
  LoomDSEMappingGenerator
  LoomRootCompleteSpatialPnrTestSupport
  LoomRuntimeDynamicWork
  MLIRArithDialect
  MLIRDataflow
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  LLVMSupport
)
