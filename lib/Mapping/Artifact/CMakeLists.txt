add_library(LoomMappingArtifact
  MappingAssembly.cpp
)

target_include_directories(LoomMappingArtifact PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomMappingArtifact PUBLIC
  LoomCommon
  MLIRMapping
)

mlir_target_link_libraries(LoomMappingArtifact PUBLIC
  MLIRIR
  MLIRParser
  MLIRSupport
)
