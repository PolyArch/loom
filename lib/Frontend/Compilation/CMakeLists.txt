add_library(LoomFrontendCompilation
  FabricCapabilityIndex.cpp
  OwnershipCandidateGenerator.cpp
  OwnershipCandidateFinalization.cpp
  OwnershipDecision.cpp
  PreMappingCompilation.cpp
  StaticGlobalMemory.cpp
  StaticMemoryBinding.cpp
  StructuredAddressIndexNarrowing.cpp
  StructuredCallSpecialization.cpp
  StructuredOwnershipAnalysis.cpp
  StructuredExecutionShape.cpp
  StructuredSpecialMathAccuracy.cpp
  StructuredMemoryChannelPromotion.cpp
  StructuredMemoryCommunication.cpp
  StructuredSchedule.cpp
  StructuredScop.cpp
  StructuredVectorScheduleVerifier.cpp
)

target_include_directories(LoomFrontendCompilation PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomFrontendCompilation PUBLIC
  LLVM
  LoomFabricIdentity
  LoomFabricMemoryOperationPort
  LoomFrontendAnalysis
  LoomFrontendLowering
  LoomFrontendRaising
  MLIRDataflow
)

mlir_target_link_libraries(LoomFrontendCompilation PUBLIC
  MLIRArithDialect
  MLIRAffineAnalysis
  MLIRAffineDialect
  MLIRAffineTransforms
  MLIRAffineToStandard
  MLIRAnalysis
  MLIRDLTIDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSCFToAffine
  MLIRTransforms
  MLIRVectorDialect
  MLIRVectorTransforms
)
