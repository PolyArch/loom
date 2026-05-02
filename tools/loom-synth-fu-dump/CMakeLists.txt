get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(loom-synth-fu-dump loom-synth-fu-dump.cpp)
llvm_update_compile_flags(loom-synth-fu-dump)

target_link_libraries(loom-synth-fu-dump PRIVATE
  MLIRFabric
  MLIRFabricTech
  MLIRFabricTechSynthesizer
  MLIRDataflow
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  ${dialect_libs}
)
