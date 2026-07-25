add_mlir_library(LoomFrontendRaising
  LiftCFToSCFPass.cpp
  LLVMArithToArithPass.cpp
  LLVMCfToCfPass.cpp
  MaterializeFMulAddPass.cpp
  NormalizeLiftedSCFExitPass.cpp
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
  MLIRFunctionInterfaces
  MLIRIR
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRRewrite
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSupport
  MLIRTransformUtils
  MLIRTransforms
  MLIRUBDialect
)
