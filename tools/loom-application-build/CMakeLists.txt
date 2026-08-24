set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Support
  TargetParser
)

add_llvm_executable(loom-application-build loom-application-build.cpp)
llvm_update_compile_flags(loom-application-build)

target_link_options(loom-application-build PRIVATE
  "LINKER:--no-as-needed,$<TARGET_FILE:LLVM>,--as-needed"
)

set_target_properties(loom-application-build PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

target_link_libraries(loom-application-build PRIVATE
  LoomApplicationBuild
  LLVMSupport
)

mlir_check_all_link_libraries(loom-application-build)
