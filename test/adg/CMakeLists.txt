add_executable(loom-adg-builder-api-test
  ADGBuiltinTest.cpp
  ADGBuilderTest.cpp
  ADGBuilderTestMain.cpp
  ADGTopologyTest.cpp
)

target_link_libraries(loom-adg-builder-api-test PRIVATE
  LoomADGBuilder
  LoomCommon
  LoomFabricArtifact
  LoomFrontendCompilation
  LLVMSupport
)
