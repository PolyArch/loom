set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_executable(loom-application-manifest-inspect
  loom-application-manifest-inspect.cpp
)
llvm_update_compile_flags(loom-application-manifest-inspect)

set_target_properties(loom-application-manifest-inspect PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_link_libraries(loom-application-manifest-inspect PRIVATE
  LoomApplicationPortfolio
  LLVMSupport
)

mlir_check_all_link_libraries(loom-application-manifest-inspect)
