add_llvm_executable(loom-mapping-estimate
  loom-mapping-estimate.cpp
)

llvm_update_compile_flags(loom-mapping-estimate)

target_link_libraries(loom-mapping-estimate
  PRIVATE
  LoomPnR
  MLIRSupport
  LLVMSupport
)

mlir_check_all_link_libraries(loom-mapping-estimate)
