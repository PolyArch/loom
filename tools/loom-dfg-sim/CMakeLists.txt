add_llvm_executable(loom-dfg-sim
  loom-dfg-sim.cpp
)

llvm_update_compile_flags(loom-dfg-sim)

target_link_libraries(loom-dfg-sim
  PRIVATE
  LoomSimulator
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRDataflow
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRParser
  MLIRSCFDialect
  MLIRSupport
  MLIRUBDialect
)

mlir_check_all_link_libraries(loom-dfg-sim)
