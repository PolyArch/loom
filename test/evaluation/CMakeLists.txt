add_executable(loom-evaluation-case-test
  CaseTest.cpp
)

target_link_libraries(loom-evaluation-case-test PRIVATE
  LoomEvaluation
  LoomImplementationPlatform
  LLVMSupport
)

add_executable(loom-evaluation-metric-test
  MetricTest.cpp
)

target_link_libraries(loom-evaluation-metric-test PRIVATE
  LoomEvaluation
  LLVMSupport
)

add_executable(loom-evaluation-tool-runner-test
  ToolRunnerTest.cpp
)

target_link_libraries(loom-evaluation-tool-runner-test PRIVATE
  LoomEvaluation
  LLVMSupport
)
