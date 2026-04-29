get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(loom-cost-test loom-cost-test.cpp)
llvm_update_compile_flags(loom-cost-test)

target_link_libraries(loom-cost-test PRIVATE
  LoomCommon
  MLIRFabric
  MLIRFabricTechSynthesizer
  MLIRDataflow
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  ${dialect_libs}
)
