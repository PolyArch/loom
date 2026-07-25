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

add_executable(loom-dataflow-canonical-reference-test
  CanonicalReferenceTest.cpp
)

target_link_libraries(loom-dataflow-canonical-reference-test PRIVATE
  LoomCommon
  MLIRDataflow
  MLIRIR
  MLIRSupport
  LLVMSupport
)
