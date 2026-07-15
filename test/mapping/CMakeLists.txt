add_executable(loom-mapping-core-test
  MappingCoreTest.cpp
)

target_link_libraries(loom-mapping-core-test PRIVATE
  LoomMapping
  LLVMSupport
)
