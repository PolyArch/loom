add_executable(loom-deployment-executable-leaves-test
  ExecutableLeavesTest.cpp
)

target_link_libraries(loom-deployment-executable-leaves-test PRIVATE
  LoomADGBuilder
  LoomDeployment
  LoomFrontendExecutable
  MLIRArithDialect
  MLIRDataflow
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRParser
  MLIRSupport
)

target_compile_definitions(loom-deployment-executable-leaves-test PRIVATE
  LOOM_TEST_LLD_PATH="${LLVM_TOOLS_BINARY_DIR}/ld.lld"
)

add_library(LoomDeploymentTestSupport STATIC
  DeploymentTestSupport.cpp
)

target_include_directories(LoomDeploymentTestSupport PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(LoomDeploymentTestSupport PUBLIC
  LoomADGBuilder
  LoomConfig
  LoomDeployment
  LoomFrontendExecutable
  LoomHardwareConfiguration
  LoomHardwareImplementation
  LoomMappingArtifact
  LoomPnR
  LoomPnRGenerator
  LoomRuntimeDeploymentLoader
  LoomRuntimePlatformBinding
  MLIRArithDialect
  MLIRDataflow
  MLIRFabric
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMapping
  MLIRMemRefDialect
  MLIRParser
  MLIRSupport
)

target_compile_definitions(LoomDeploymentTestSupport PRIVATE
  LOOM_TEST_LLD_PATH="${LLVM_TOOLS_BINARY_DIR}/ld.lld"
)

add_executable(loom-deployment-closure-test
  DeploymentClosureTest.cpp
)

target_link_libraries(loom-deployment-closure-test PRIVATE
  LoomDSE
  LoomDeploymentTestSupport
  LoomRuntimeDeploymentLoader
  LoomRuntimeResourceTime
)

add_executable(loom-deployment-package-test
  DeploymentPackageTest.cpp
)

target_link_libraries(loom-deployment-package-test PRIVATE
  LoomDeploymentTestSupport
)
