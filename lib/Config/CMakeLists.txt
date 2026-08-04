add_library(LoomConfig
  ResolvedConfig.cpp
)

target_include_directories(LoomConfig PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomConfig PUBLIC
  LoomCommon
  LoomDSEPolicy
  LLVMSupport
)
