add_llvm_executable(loom-adg-builder-test
  loom-adg-builder-test.cpp
)

llvm_update_compile_flags(loom-adg-builder-test)

target_link_libraries(loom-adg-builder-test
  PRIVATE
  LoomADGBuilder
  LLVMSupport
)

mlir_check_all_link_libraries(loom-adg-builder-test)
