set(LLVM_LINK_COMPONENTS
  BitReader
  Support
)

add_llvm_executable(loom-payload loom-payload.cpp)
llvm_update_compile_flags(loom-payload)

set_target_properties(loom-payload PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

target_link_libraries(loom-payload PRIVATE
  LoomFrontendPayload
)

mlir_check_all_link_libraries(loom-payload)
