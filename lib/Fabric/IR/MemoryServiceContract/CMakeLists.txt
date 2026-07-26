add_library(LoomFabricMemoryServiceContract
  MemoryServiceContract.cpp
  MemoryServiceContractCodec.cpp
)

target_include_directories(LoomFabricMemoryServiceContract PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomFabricMemoryServiceContract PUBLIC
  LoomFabricIdentity
  LoomFabricMemoryCapabilityDomains
  LoomFabricMemoryConsistencyContract
  LoomFabricResourceContract
  MLIRDataflow
  MLIRIR
  MLIRSupport
  LLVMSupport
)
