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

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Frontend/Raising

)

target_link_libraries(LoomFrontendRaising PUBLIC
  LoomFrontendAnalysis
  LoomFrontendLowering
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

add_llvm_pass_plugin(LoomCandidateProjectionPass
  CandidateProjectionPass.cpp

  PARTIAL_SOURCES_INTENDED
)

target_link_libraries(LoomCandidateProjectionPass PRIVATE
  LoomFrontendPayload
  LoomFrontendRaising
)

set_target_properties(LoomCandidateProjectionPass PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/loom"
)
