add_library(LoomMappingInspection
  SpatialMappingInspection.cpp
)

target_include_directories(LoomMappingInspection PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomMappingInspection PUBLIC
  LoomMappingArtifact
  LoomFabricIdentity
  LLVMSupport
)
