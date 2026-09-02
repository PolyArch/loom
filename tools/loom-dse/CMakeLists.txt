get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(loom-dse loom-dse.cpp)
llvm_update_compile_flags(loom-dse)

target_link_libraries(loom-dse PRIVATE
  LoomDSE
  LoomDSEProductionOwners
  LoomDSECompilerGenerator
  LoomDSEFeedback
  LoomDSEFpaCampaign
  LoomDSEHardwareGenerator
  LoomDSEMappingGenerator
  LoomDSEModelingGenerator
  LoomEDAOpenSource
  MLIRDataflow
  MLIRIR
  MLIRParser
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  ${dialect_libs}
  ${conversion_libs}
  LLVMSupport
)

mlir_check_all_link_libraries(loom-dse)
