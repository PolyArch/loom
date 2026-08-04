add_executable(loom-implementation-platform-test
  ImplementationPlatformTest.cpp
)

target_link_libraries(loom-implementation-platform-test PRIVATE
  LoomImplementationPlatform
  LLVMSupport
)
