get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(loom-synth-verifier-test loom-synth-verifier-test.cpp)
llvm_update_compile_flags(loom-synth-verifier-test)

target_link_libraries(loom-synth-verifier-test PRIVATE
  MLIRFabric
  MLIRFabricTech
  MLIRFabricTechSynthesizer
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  ${dialect_libs}
)
