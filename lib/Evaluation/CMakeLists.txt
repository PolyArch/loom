add_library(LoomEvaluation
  ArtifactImportCache.cpp
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
  ModelParameterBundle.cpp
  ModelParameterContract.cpp
  ModelProvider.cpp
  NumericValue.cpp
  OwnerError.cpp
  PhysicalRailAnalysis.cpp
  QueryText.cpp
  Request.cpp
  RequestText.cpp
  Scope.cpp
  StandardFindings.cpp
)

target_include_directories(LoomEvaluation PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomEvaluation PUBLIC
  LoomCommon
  LoomExternalTool
  LoomImplementationPlatform
  LLVMSupport
)
