add_library(LoomConfig
  DseConfigProjection.cpp
  ResolvedConfig.cpp
  ResolvedConfigJson.cpp
)

target_include_directories(LoomConfig PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomConfig PUBLIC
  LoomCommon
  LoomDSEPolicy
  MLIRFabricEnums
  LLVMSupport
)
