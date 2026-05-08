add_mlir_library(LoomFrontendRaising
  LLVMArithToArithPass.cpp
  LLVMCfToCfPass.cpp
  LLVMFuncToFuncPass.cpp
  Pipeline.cpp
  SCFForToForallPass.cpp
  SCFWhileToForPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Frontend/Raising

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRControlFlowToSCF
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSupport
  MLIRTransforms
)
