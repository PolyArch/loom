add_executable(loom-fast-fpa-model-test
  FastFpaModelTest.cpp
)

target_link_libraries(loom-fast-fpa-model-test PRIVATE
  LoomDSEHardwareGenerator
  LoomStructuredFabricAnalyticModel
  LLVMSupport
)
