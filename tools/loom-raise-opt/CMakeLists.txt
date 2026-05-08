get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(loom-raise-opt loom-raise-opt.cpp)
llvm_update_compile_flags(loom-raise-opt)

set_target_properties(loom-raise-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

target_link_libraries(loom-raise-opt PRIVATE
  LoomFrontendRaising
  MLIRIR
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIRSupport
  ${dialect_libs}
  ${conversion_libs}
)

mlir_check_all_link_libraries(loom-raise-opt)
