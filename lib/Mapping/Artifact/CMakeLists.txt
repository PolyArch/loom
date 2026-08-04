add_library(LoomMappingArtifact
  MappingAssembly.cpp
  MappingConstraintSet.cpp
  SpatialMappingCapacityVerification.cpp
  SpatialMappingArtifact.cpp
  SpatialProgressAnalysis.cpp
  SpatialMappingTagAssignments.cpp
  SpatialMappingMemoryImport.cpp
  TechMappingCanonicalKey.cpp
  TechMappingComputeVerification.cpp
  TechMappingArtifact.cpp
  TechMappingMemoryProjection.cpp
)

target_include_directories(LoomMappingArtifact PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomMappingArtifact PUBLIC
  LoomCommon
  LoomFabricArtifact
  MLIRDataflow
  MLIRMapping
)

mlir_target_link_libraries(LoomMappingArtifact PUBLIC
  MLIRIR
  MLIRParser
  MLIRSupport
)
