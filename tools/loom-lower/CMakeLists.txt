set(LLVM_LINK_COMPONENTS
  Support
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(loom-lower loom-lower.cpp)
llvm_update_compile_flags(loom-lower)

# Keep the binary in <build>/bin for source-to-IR integration and lit
# substitutions next to loom-cc / loom-raise.
set_target_properties(loom-lower PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

target_link_libraries(loom-lower PRIVATE
  LoomFrontendLowering
  MLIRDataflow
  MLIRLoom
)

mlir_target_link_libraries(loom-lower PRIVATE
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRParser
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRSCFDialect
  MLIRSupport
  MLIRTransforms
  MLIRUBDialect
  ${dialect_libs}
  ${conversion_libs}
)

mlir_check_all_link_libraries(loom-lower)
