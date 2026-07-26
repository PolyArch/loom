add_library(LoomFabricFuCapabilityDomain
  FuCapabilityDomain.cpp
)

target_include_directories(LoomFabricFuCapabilityDomain PUBLIC
  ${LOOM_SOURCE_DIR}/include
)

target_link_libraries(LoomFabricFuCapabilityDomain PUBLIC
  LLVMSupport
)
