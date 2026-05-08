add_mlir_library(LoomFrontendRaising
  LLVMArithToArithPass.cpp
  LLVMCfToCfPass.cpp
  LLVMFuncToFuncPass.cpp
  Pipeline.cpp
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
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSupport
  MLIRTransforms
)
