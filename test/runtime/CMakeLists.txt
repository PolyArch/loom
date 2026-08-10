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
