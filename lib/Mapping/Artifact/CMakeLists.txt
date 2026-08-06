add_library(LoomMappingArtifact
  ConfiguredHardwareProjection.cpp
  MappingAssembly.cpp
  MappingConstraintSet.cpp
  MappingConstraintSetBuilder.cpp
  SpatialMappingConstraintAdmission.cpp
  SpatialMappingCapacityVerification.cpp
  SpatialMappingArtifact.cpp
  SpatialProgressAnalysis.cpp
  SpatialMappingTagAssignments.cpp
  SpatialMappingMemoryImport.cpp
  SystemMappingConstraintSet.cpp
  SystemMappingConstraintSetBuilder.cpp
  SystemMappingArtifact.cpp
  SystemMappingIdentity.cpp
  SystemPresburger.cpp
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
  MLIRPresburger
  MLIRSupport
)
