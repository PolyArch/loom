add_mlir_library(LoomFrontendRaising
  CandidateHints.cpp
  ConstantCallbackSpecialization.cpp
  CountedLoopProjection.cpp
  DeduplicateSCFWhileStatePass.cpp
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

target_link_libraries(LoomFrontendRaising PUBLIC
  LoomFrontendAnalysis
  MLIRLoom
)
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
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIRRewrite
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSupport
  MLIRTargetLLVMIRImport
  MLIRTargetLLVMIRExport
  MLIRToLLVMIRTranslationRegistration
  MLIRTransformUtils
  MLIRTransforms
  MLIRUBDialect
  LLVMCore
  LLVMAnalysis
  LLVMipo
  LLVMPasses
  LLVMTransformUtils
)
