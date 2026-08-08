add_executable(loom-yosys-synthesis-test
  YosysSynthesisTest.cpp
)

target_link_libraries(loom-yosys-synthesis-test PRIVATE
  LoomEDAOpenSource
  LLVMSupport
)

add_executable(loom-verilator-functional-test
  VerilatorFunctionalTest.cpp
)

target_link_libraries(loom-verilator-functional-test PRIVATE
  LoomEDAOpenSource
  LLVMSupport
)

get_property(LOOM_EDA_TEST_TARGETS
  DIRECTORY
  PROPERTY BUILDSYSTEM_TARGETS
)
set(LOOM_EDA_TEST_TARGETS
  ${LOOM_EDA_TEST_TARGETS}
  PARENT_SCOPE
)
