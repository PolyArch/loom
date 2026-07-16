add_executable(loom-evaluation-metric-test
  MetricTest.cpp
)

target_link_libraries(loom-evaluation-metric-test PRIVATE
  LoomEvaluation
  LLVMSupport
)
