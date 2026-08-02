add_subdirectory(IR)
add_subdirectory(Artifact)

add_library(LoomMapping
  ComputeCapabilityVerification.cpp
  FabricOccurrenceIndex.cpp
  FabricRoutingIndex.cpp
  MemoryRealizationProjection.cpp
  Verifier.cpp
)

target_include_directories(LoomMapping PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomMapping PUBLIC
  LoomCommon
  LoomFabricArtifact
  LoomFabricBoundaryDataPath
  LoomFabricImplementationFamily
  LLVMSupport
)
