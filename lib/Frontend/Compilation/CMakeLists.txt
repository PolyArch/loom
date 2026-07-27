add_library(LoomFrontendCompilation
  PreMappingCompilation.cpp
)

target_include_directories(LoomFrontendCompilation PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomFrontendCompilation PUBLIC
  LoomFrontendLowering
  LoomFrontendRaising
  MLIRDataflow
  MLIRIR
  LLVMCore
  LLVMSupport
)
