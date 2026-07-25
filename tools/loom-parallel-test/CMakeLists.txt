add_llvm_executable(loom-parallel-test loom-parallel-test.cpp)
llvm_update_compile_flags(loom-parallel-test)

target_link_libraries(loom-parallel-test PRIVATE
  MLIRFabricTech
  LoomCommon
  LLVMSupport
)
