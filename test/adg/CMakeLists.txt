add_executable(loom-adg-builder-api-test
  ADGBuiltinTest.cpp
  ADGBuilderTest.cpp
  ADGBuilderTestMain.cpp
  ADGServiceLegCarrierTest.cpp
  ADGTopologyTest.cpp
  BuilderDomainAuthoringTest.cpp
)

target_include_directories(loom-adg-builder-api-test PRIVATE
  ${LOOM_SOURCE_DIR}/lib/Fabric/Artifact
)

target_link_libraries(loom-adg-builder-api-test PRIVATE
  LoomADGBuilder
  LoomCommon
  LoomFabricArtifact
  LoomFrontendCompilation
  LLVMSupport
)
