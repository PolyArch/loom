add_llvm_executable(loom-dfg-sim
  loom-dfg-sim.cpp
)

llvm_update_compile_flags(loom-dfg-sim)

target_link_libraries(loom-dfg-sim
  PRIVATE
  LoomSimulator
  MLIRArithDialect
  MLIRDataflow
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRTransforms
)

mlir_check_all_link_libraries(loom-dfg-sim)
