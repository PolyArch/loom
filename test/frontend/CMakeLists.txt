add_subdirectory(payload)

add_executable(loom-structured-program-artifact-test
  StructuredProgramArtifactTest.cpp
)

target_link_libraries(loom-structured-program-artifact-test PRIVATE
  LoomCommon
  LoomFrontendLowering
  MLIRLoom
  MLIRArithDialect
  MLIRDataflow
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRParser
  MLIRSupport
  LLVMSupport
)

add_executable(loom-pre-mapping-compilation-test
  PreMappingCompilationTest.cpp
)

target_link_libraries(loom-pre-mapping-compilation-test PRIVATE
  LoomADGBuilder
  LoomCommon
  LoomFrontendCompilation
  LLVMCore
  LLVMIRReader
  LLVMSupport
)
