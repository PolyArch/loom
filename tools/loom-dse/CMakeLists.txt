add_llvm_executable(loom-dse loom-dse.cpp)
llvm_update_compile_flags(loom-dse)

target_link_libraries(loom-dse PRIVATE
  LoomDSE
  LoomDSECompilerGenerator
  LoomDSEFeedback
  LoomDSEHardwareGenerator
  LoomDSEMappingGenerator
  LoomDSEModelingGenerator
  LLVMSupport
)
