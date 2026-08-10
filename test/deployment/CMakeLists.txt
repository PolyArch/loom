add_executable(loom-deployment-executable-leaves-test
  ExecutableLeavesTest.cpp
)

target_link_libraries(loom-deployment-executable-leaves-test PRIVATE
  LoomADGBuilder
  LoomDeployment
  LoomFrontendExecutable
  MLIRArithDialect
  MLIRDataflow
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRParser
  MLIRSupport
)

target_compile_definitions(loom-deployment-executable-leaves-test PRIVATE
  LOOM_TEST_LLD_PATH="${LLVM_TOOLS_BINARY_DIR}/ld.lld"
)
