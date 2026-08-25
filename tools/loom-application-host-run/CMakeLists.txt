set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_executable(loom-application-host-run
  loom-application-host-run.cpp
)
llvm_update_compile_flags(loom-application-host-run)

set_target_properties(loom-application-host-run PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_link_libraries(loom-application-host-run PRIVATE
  LoomApplicationHostRunner
  LLVMSupport
)

mlir_check_all_link_libraries(loom-application-host-run)
