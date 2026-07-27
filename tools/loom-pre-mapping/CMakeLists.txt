set(LLVM_LINK_COMPONENTS
  AsmParser
  BitReader
  Core
  IRReader
  Support
)

add_llvm_executable(loom-pre-mapping loom-pre-mapping.cpp)
llvm_update_compile_flags(loom-pre-mapping)

# Keep the binary in <build>/bin next to loom-cc / loom-c++ / loom-raise so
# the corpus gate can resolve it like the other production tools.
set_target_properties(loom-pre-mapping PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

target_link_libraries(loom-pre-mapping PRIVATE
  LoomADGBuilder
  LoomCommon
  LoomFrontendCompilation
)

mlir_target_link_libraries(loom-pre-mapping PRIVATE
  MLIRIR
  MLIRSupport
)

mlir_check_all_link_libraries(loom-pre-mapping)
