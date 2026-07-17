add_executable(loom-mapping-core-test
  MappingComputeFreezeTest.cpp
  MappingCoreTest.cpp
  MappingCoreTestSupport.cpp
  MappingMemoryTest.cpp
  MappingVerifierTest.cpp
)

target_link_libraries(loom-mapping-core-test PRIVATE
  LoomMapping
  LoomPnR
  LLVMSupport
)
