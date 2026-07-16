get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(loom-synth-base-test loom-synth-base-test.cpp)
llvm_update_compile_flags(loom-synth-base-test)

target_link_libraries(loom-synth-base-test PRIVATE
  LoomCommon
  MLIRArithDialect
  MLIRDataflow
  MLIRFabric
  MLIRFabricTech
  MLIRFabricTechSynthesizer
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  ${dialect_libs}
)
