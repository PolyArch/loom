add_executable(loom-eda-open-source-adapter-test
  OpenSourceAdapterTest.cpp
)

target_link_libraries(loom-eda-open-source-adapter-test PRIVATE
  LoomEDAOpenSource
  LoomExternalTool
  LLVMSupport
)

add_executable(loom-yosys-synthesis-test
  YosysSynthesisTest.cpp
)

target_link_libraries(loom-yosys-synthesis-test PRIVATE
  LoomEDAOpenSource
  LLVMSupport
)
