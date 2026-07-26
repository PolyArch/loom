add_library(LoomFabricMemoryCapabilityDomains
  MemoryActorContractDomain.cpp
  MemoryCapabilityRelation.cpp
  MemoryCapabilityDomains.cpp
  ReducedProductRelation.cpp
)

target_include_directories(LoomFabricMemoryCapabilityDomains PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomFabricMemoryCapabilityDomains PUBLIC
  MLIRDataflow
  LLVMSupport
)
