add_subdirectory(payload)

add_executable(loom-structured-program-artifact-test
  StructuredProgramArtifactTest.cpp
)

target_link_libraries(loom-structured-program-artifact-test PRIVATE
  LoomCommon
  MLIRLoom
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRParser
  MLIRSupport
  LLVMSupport
)
