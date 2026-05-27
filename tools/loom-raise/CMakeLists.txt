set(LLVM_LINK_COMPONENTS
  AsmParser
  BitReader
  Core
  IRReader
  Support
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(loom-raise loom-raise.cpp)
llvm_update_compile_flags(loom-raise)

# Drop the binary into <build>/bin so test/app/raise_check.sh and the lit
# substitution can locate it next to loom-cc / loom-c++.
set_target_properties(loom-raise PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

target_link_libraries(loom-raise PRIVATE
  LoomFrontendRaising
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRControlFlowToSCF
  MLIRFromLLVMIRTranslationRegistration
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRLLVMIRTransforms
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRParser
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIRSCFDialect
  MLIRSupport
  MLIRTargetLLVMIRImport
  MLIRTransforms
  MLIRUBDialect
  ${dialect_libs}
  ${conversion_libs}
  ${translation_libs}
)

mlir_check_all_link_libraries(loom-raise)
