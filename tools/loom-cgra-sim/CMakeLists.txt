add_llvm_executable(loom-cgra-sim
  loom-cgra-sim.cpp
)

llvm_update_compile_flags(loom-cgra-sim)

target_link_libraries(loom-cgra-sim
  PRIVATE
  LoomSimulator
  MLIRSupport
  LLVMSupport
)

mlir_check_all_link_libraries(loom-cgra-sim)
