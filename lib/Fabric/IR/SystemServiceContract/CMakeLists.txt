add_library(LoomFabricSystemServiceContract
  SystemServiceContract.cpp
  SystemServiceContractCodec.cpp
)

target_include_directories(LoomFabricSystemServiceContract PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomFabricSystemServiceContract PUBLIC
  LoomFabricIdentity
  LoomFabricMemoryCapabilityDomains
  LoomFabricMemoryConsistencyContract
  MLIRDataflow
  MLIRIR
  MLIRSupport
  LLVMSupport
)
