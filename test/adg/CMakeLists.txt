add_executable(loom-adg-builder-api-test
  ADGBuilderTest.cpp
)

target_link_libraries(loom-adg-builder-api-test PRIVATE
  LoomADGBuilder
  LoomCommon
  LoomFabricArtifact
  LoomFrontendCompilation
  LLVMSupport
)
