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

add_executable(loom-dataflow-service-schema-test
  DataflowServiceSchemaTest.cpp
)

target_link_libraries(loom-dataflow-service-schema-test PRIVATE
  LoomCommon
  MLIRDLTIDialect
  MLIRDataflow
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
)

add_executable(loom-dataflow-operation-schema-test
  OperationSchemaTest.cpp
)

target_link_libraries(loom-dataflow-operation-schema-test PRIVATE
  MLIRDataflow
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRUBDialect
  MLIRSupport
  LLVMSupport
)
