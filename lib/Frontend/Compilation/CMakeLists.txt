add_library(LoomFrontendCompilation
  FabricCapabilityIndex.cpp
  OwnershipCandidateGenerator.cpp
  PreMappingCompilation.cpp
  StaticGlobalMemory.cpp
  StaticMemoryBinding.cpp
  StructuredAddressIndexNarrowing.cpp
  StructuredCallSpecialization.cpp
  StructuredExecutionShape.cpp
  StructuredSchedule.cpp
)

target_include_directories(LoomFrontendCompilation PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomFrontendCompilation PUBLIC
  LLVM
  LoomFabricIdentity
  LoomFabricMemoryOperationPort
  LoomFrontendLowering
  LoomFrontendRaising
  MLIRDataflow
)

mlir_target_link_libraries(LoomFrontendCompilation PUBLIC
  MLIRArithDialect
  MLIRDLTIDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTransforms
)
