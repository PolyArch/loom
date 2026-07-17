find_package(Threads REQUIRED)

add_library(LoomEvaluation
  Metric.cpp
  ToolRunner.cpp
)

target_include_directories(LoomEvaluation PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomEvaluation PUBLIC
  LoomCommon
  LLVMSupport
  Threads::Threads
)
