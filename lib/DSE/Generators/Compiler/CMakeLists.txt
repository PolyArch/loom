add_library(LoomDSECompilerGenerator
  DataflowRewriteCandidateGenerator.cpp
  PreMappingExploration.cpp
  StructuredExecutionShapeCandidateGenerator.cpp
  StructuredMemoryCommunicationCandidateGenerator.cpp
  StructuredOwnershipCandidateGenerator.cpp
  StructuredScheduleCandidateGenerator.cpp
)

target_include_directories(LoomDSECompilerGenerator PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomDSECompilerGenerator PUBLIC
  LoomDSE
  LoomFabricArtifact
  LoomFrontendCompilation
  LoomSimulator
  MLIRDataflowTransforms
  LLVMSupport
)
