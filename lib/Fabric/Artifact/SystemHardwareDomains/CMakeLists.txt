add_library(LoomFabricHardwareDomainContracts
  FabricHardwareDomainContracts.cpp
)

target_include_directories(LoomFabricHardwareDomainContracts PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomFabricHardwareDomainContracts PUBLIC
  LoomFabricIdentity
  LoomFabricMemoryConsistencyContract
  LoomFabricSystemContracts
  LLVMSupport
)
