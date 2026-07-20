add_executable(loom-simulator-vector-boundary-test
  DFGVectorBoundaryTest.cpp
)

target_include_directories(loom-simulator-vector-boundary-test PRIVATE
  ${LOOM_SOURCE_DIR}/lib/Simulator
)

target_link_libraries(loom-simulator-vector-boundary-test PRIVATE
  LoomSimulator
  MLIRArithDialect
  MLIRDataflow
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRParser
  MLIRSupport
  MLIRUBDialect
  LLVMSupport
)
