get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(loom loom.cpp)
llvm_update_compile_flags(loom)

target_link_libraries(loom PRIVATE
  MLIRFabric
  MLIRDataflow
  MLIRIR
  MLIROptLib
  MLIRParser
  MLIRSupport
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRRegisterAllExtensions
  ${dialect_libs}
  ${conversion_libs}
)

mlir_check_all_link_libraries(loom)
