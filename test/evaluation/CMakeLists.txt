add_executable(loom-evaluation-case-test
  CaseTest.cpp
)

target_include_directories(loom-evaluation-case-test PRIVATE
  ${PROJECT_SOURCE_DIR}/lib/Common
)

target_link_libraries(loom-evaluation-case-test PRIVATE
  LoomConfig
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

add_executable(loom-evaluation-evidence-test
  EvidenceTest.cpp
)

target_link_libraries(loom-evaluation-evidence-test PRIVATE
  LoomConfig
  LoomEvaluation
  LoomExternalTool
  LLVMSupport
)
