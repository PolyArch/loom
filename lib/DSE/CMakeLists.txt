add_library(LoomDSE
  Promotion.cpp
  StructuredOwnership.cpp
)

target_include_directories(LoomDSE PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomDSE PUBLIC
  LoomCommon
  LoomEvaluation
  LoomFrontendCompilation
  LoomStructuredFabricAnalyticModel
  LLVMSupport
)
