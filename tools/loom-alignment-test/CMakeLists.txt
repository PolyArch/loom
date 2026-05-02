get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(loom-alignment-test loom-alignment-test.cpp)
llvm_update_compile_flags(loom-alignment-test)

target_link_libraries(loom-alignment-test PRIVATE
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
