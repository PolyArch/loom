get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(loom-template-dump loom-template-dump.cpp)
llvm_update_compile_flags(loom-template-dump)

target_link_libraries(loom-template-dump PRIVATE
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
