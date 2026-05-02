get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(loom-coverage-test loom-coverage-test.cpp)
llvm_update_compile_flags(loom-coverage-test)

target_link_libraries(loom-coverage-test PRIVATE
  LoomCommon
  MLIRDataflow
  MLIRFabric
  MLIRFabricTech
  MLIRFabricTechSynthesizer
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  ${dialect_libs}
)
