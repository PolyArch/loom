add_llvm_executable(loom-synth-config-test loom-synth-config-test.cpp)
llvm_update_compile_flags(loom-synth-config-test)

target_link_libraries(loom-synth-config-test PRIVATE
  LoomCommon
  LLVMSupport
)
