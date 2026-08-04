add_library(LoomDSECompilerGenerator
  StructuredOwnershipCandidateGenerator.cpp
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
