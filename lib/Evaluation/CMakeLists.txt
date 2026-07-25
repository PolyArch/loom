find_package(Threads REQUIRED)

add_library(LoomEvaluation
  CanonicalSupport.cpp
  CaseSignature.cpp
  CaseText.cpp
  Condition.cpp
  Metric.cpp
  ModelDescriptor.cpp
  NumericValue.cpp
  Request.cpp
  Scope.cpp
  ToolRunner.cpp
)

target_include_directories(LoomEvaluation PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomEvaluation PUBLIC
  LoomCommon
  LoomImplementationPlatform
  LLVMSupport
  Threads::Threads
)
