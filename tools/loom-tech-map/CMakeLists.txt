set(LLVM_LINK_COMPONENTS
  Support
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(loom-tech-map
  loom-tech-map.cpp
)

llvm_update_compile_flags(loom-tech-map)

set_target_properties(loom-tech-map PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_link_libraries(loom-tech-map PRIVATE
  LoomCommon
  LoomTechMapping
  MLIRDataflow
)

mlir_target_link_libraries(loom-tech-map PRIVATE
  MLIRIR
  MLIRParser
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRSupport
  ${dialect_libs}
  ${conversion_libs}
)

mlir_check_all_link_libraries(loom-tech-map)
