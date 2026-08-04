add_library(LoomDSEMappingGenerator
  SpatialPnrCandidateGenerator.cpp
)

target_include_directories(LoomDSEMappingGenerator PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomDSEMappingGenerator PUBLIC
  LoomDSECandidateGenerator
  LoomFabricArtifact
  LoomMappingArtifact
  LoomPnRGenerator
  MLIRDataflow
  LLVMSupport
)
