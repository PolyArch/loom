add_library(LoomTechMapping
  TechMappingConfig.cpp
)

target_include_directories(LoomTechMapping PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomTechMapping PUBLIC
  LoomCommon
  LLVMSupport
)
