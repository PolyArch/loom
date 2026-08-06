add_library(LoomSystemMappingIdentity
  SystemMappingIdentity.cpp
)

target_include_directories(LoomSystemMappingIdentity PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomSystemMappingIdentity PUBLIC
  LoomCommon
  MLIRDataflow
  LLVMSupport
)
