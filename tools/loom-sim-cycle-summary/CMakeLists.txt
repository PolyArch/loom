add_llvm_executable(loom-sim-cycle-summary loom-sim-cycle-summary.cpp)
llvm_update_compile_flags(loom-sim-cycle-summary)

target_link_libraries(loom-sim-cycle-summary PRIVATE
  LoomSimulator
  LLVMSupport
)
