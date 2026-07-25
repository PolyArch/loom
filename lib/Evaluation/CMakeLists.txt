find_package(Threads REQUIRED)

add_library(LoomEvaluation
  CanonicalSupport.cpp
  CaseSignature.cpp
  CaseText.cpp
  Condition.cpp
  ConditionText.cpp
  Evidence.cpp
  EvidenceText.cpp
  Finding.cpp
  Metric.cpp
  ModelDescriptor.cpp
  NumericValue.cpp
  OwnerError.cpp
  QueryText.cpp
  Request.cpp
  RequestText.cpp
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
