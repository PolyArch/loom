add_library(LoomTechMapping
  TechMappingCandidateDomain.cpp
  TechMappingComputeRows.cpp
  TechMappingConfig.cpp
  TechMappingGenerator.cpp
  TechMappingMaterialization.cpp
  TechMappingMemoryRows.cpp
  TechMappingSearch.cpp
)

target_include_directories(LoomTechMapping PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomTechMapping PUBLIC
  LoomCommon
  LoomMappingArtifact
  LoomFabricArtifact
  MLIRDataflow
  MLIRMapping
  LLVMSupport
)

mlir_target_link_libraries(LoomTechMapping PUBLIC
  MLIRIR
  MLIRSupport
)
