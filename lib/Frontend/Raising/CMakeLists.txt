add_mlir_library(LoomFrontendRaising
  LiftCFToSCFPass.cpp
  LLVMArithToArithPass.cpp
  LLVMCfToCfPass.cpp
  MaterializeFMulAddPass.cpp
  NormalizeLiftedSCFExitPass.cpp
  Pipeline.cpp
  SCFForToForallPass.cpp
  SCFWhileToForPass.cpp
  StructuredRaising.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Frontend/Raising

)

target_link_libraries(LoomFrontendRaising PUBLIC MLIRLoom)
mlir_target_link_libraries(LoomFrontendRaising PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRControlFlowToSCF
  MLIRFuncDialect
  MLIRFunctionInterfaces
  MLIRIR
  MLIRFromLLVMIRTranslationRegistration
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRRewrite
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSupport
  MLIRTargetLLVMIRImport
  MLIRTransformUtils
  MLIRTransforms
  MLIRUBDialect
  LLVMCore
)
