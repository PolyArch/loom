add_llvm_executable(loom-application-build loom-application-build.cpp)
llvm_update_compile_flags(loom-application-build)

set_target_properties(loom-application-build PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

target_link_libraries(loom-application-build PRIVATE
  LoomADGBuilder
  LoomApplicationBuild
  LoomDeployment
  LoomFabricIdentity
  LoomFrontendExecutable
  LoomFrontendPayload
  LLVMSupport
)

target_compile_definitions(loom-application-build PRIVATE
  LOOM_LLVM_TOOLS_DIR="${LLVM_TOOLS_BINARY_DIR}"
)

mlir_check_all_link_libraries(loom-application-build)
