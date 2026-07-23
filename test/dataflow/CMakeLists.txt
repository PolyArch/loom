add_executable(loom-dataflow-memory-effect-test
  DataflowMemoryEffectTest.cpp
)

target_link_libraries(loom-dataflow-memory-effect-test PRIVATE
  MLIRDataflow
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRSupport
  LLVMSupport
)
