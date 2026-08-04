add_library(LoomDSECompilerGenerator
  PreMappingExploration.cpp
  StructuredExecutionShapeCandidateGenerator.cpp
  StructuredOwnershipCandidateGenerator.cpp
  StructuredScheduleCandidateGenerator.cpp
)

target_include_directories(LoomDSECompilerGenerator PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomDSECompilerGenerator PUBLIC
  LoomDSE
  LoomFabricArtifact
  LoomSimulator
  LLVMSupport
)
