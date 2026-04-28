get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(loom-candidate-dump loom-candidate-dump.cpp)
llvm_update_compile_flags(loom-candidate-dump)

target_link_libraries(loom-candidate-dump PRIVATE
  MLIRFabric
  MLIRFabricTech
  MLIRDataflow
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  ${dialect_libs}
)
