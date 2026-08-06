add_library(LoomEDAOpenSource
  Yosys.cpp
)

target_include_directories(LoomEDAOpenSource PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomEDAOpenSource PUBLIC
  LoomExternalTool
  LLVMSupport
)
