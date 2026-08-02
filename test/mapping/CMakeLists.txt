add_executable(loom-mapping-core-test
  MappingCapabilityTemplateTest.cpp
  MappingComputeFreezeTest.cpp
  MappingCoreTest.cpp
  MappingCoreTestSupport.cpp
  MappingMemoryTest.cpp
  MappingRoutingFreezeTest.cpp
  MappingVerifierTest.cpp
  PnrProblemInputsTest.cpp
)

add_executable(loom-mapping-artifact-test
  MappingArtifactTest.cpp
)

target_link_libraries(loom-mapping-artifact-test PRIVATE
  LoomMappingArtifact
  MLIRIR
  MLIRParser
  LLVMSupport
)

target_link_libraries(loom-mapping-core-test PRIVATE
  LoomMapping
  LoomPnR
  LLVMSupport
  MLIRIR
  MLIRLLVMDialect
)
