add_library(LoomMappingArtifact
  ConfiguredHardwareBoundaryProjection.cpp
  ConfiguredHardwareMemoryProjection.cpp
  ConfiguredHardwarePeProjection.cpp
  ConfiguredHardwareProjection.cpp
  ConfiguredHardwareTagProjection.cpp
  MappingAssembly.cpp
  MappingConstraintCanonicalization.cpp
  MappingConstraintSet.cpp
  MappingConstraintSetBuilder.cpp
  MappingResourceUseImport.cpp
  ResourceCapacityVerification.cpp
  SpatialMappingConstraintAdmission.cpp
  SpatialMappingCapacityVerification.cpp
  SpatialMappingArtifact.cpp
  SpatialMappingImportContext.cpp
  MappingProgressAnalysis.cpp
  SpatialMappingTagAssignments.cpp
  SpatialMappingMemoryImport.cpp
  SpatialPhysicalDemandProjection.cpp
  SystemMappingConstraintSet.cpp
  SystemMappingConstraintSetBuilder.cpp
  SystemConfiguredHardwareProjection.cpp
  SystemMappingArtifact.cpp
  SystemMappingCapacityVerification.cpp
  SystemMappingClosure.cpp
  SystemMappingClosureProjection.cpp
  SystemMappingConstraintAdmission.cpp
  SystemMappingExecutionProjection.cpp
  SystemMappingHandshakeVerification.cpp
  SystemMappingServiceTargetVerification.cpp
  SystemServiceBindingProjection.cpp
  SystemPresburger.cpp
  TechMappingCanonicalKey.cpp
  TechMappingComputeVerification.cpp
  TechMappingArtifact.cpp
  TechMappingMemoryIngress.cpp
  TechMappingMemoryLegality.cpp
  TechMappingMemoryProjection.cpp
)

target_include_directories(LoomMappingArtifact PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomMappingArtifact PUBLIC
  LoomCommon
  LoomFabricArtifact
  LoomSystemMappingIdentity
  MLIRDataflow
  MLIRMapping
)

mlir_target_link_libraries(LoomMappingArtifact PUBLIC
  MLIRAffineAnalysis
  MLIRIR
  MLIRParser
  MLIRPresburger
  MLIRSupport
)
