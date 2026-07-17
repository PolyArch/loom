add_executable(loom-mapping-core-test
  MappingComputeFreezeTest.cpp
  MappingCoreTest.cpp
  MappingCoreTestSupport.cpp
  MappingMemoryTest.cpp
  MappingRoutingFreezeTest.cpp
  MappingVerifierTest.cpp
)

target_include_directories(loom-mapping-core-test PRIVATE
  ${LOOM_SOURCE_DIR}/lib
)

target_link_libraries(loom-mapping-core-test PRIVATE
  LoomMapping
  LoomPnR
  LLVMSupport
)
