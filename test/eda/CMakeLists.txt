add_executable(loom-yosys-synthesis-test
  YosysSynthesisTest.cpp
)

target_link_libraries(loom-yosys-synthesis-test PRIVATE
  LoomEDAOpenSource
  LLVMSupport
)
