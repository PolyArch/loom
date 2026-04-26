add_llvm_executable(loom-config-test loom-config-test.cpp)
llvm_update_compile_flags(loom-config-test)

target_link_libraries(loom-config-test PRIVATE
  LoomCommon
  LLVMSupport
)
