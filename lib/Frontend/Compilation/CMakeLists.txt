add_library(LoomFrontendCompilation
  FabricCapabilityIndex.cpp
  PreMappingCompilation.cpp
)

target_include_directories(LoomFrontendCompilation PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomFrontendCompilation PUBLIC
  LoomFabricIdentity
  LoomFrontendLowering
  LoomFrontendRaising
  MLIRDataflow
  MLIRIR
  LLVMCore
  LLVMSupport
)
