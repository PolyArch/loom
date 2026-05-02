add_llvm_executable(loom-hwsg-test loom-hwsg-test.cpp)
llvm_update_compile_flags(loom-hwsg-test)

target_link_libraries(loom-hwsg-test PRIVATE
  LoomCommon
  LLVMSupport
)
