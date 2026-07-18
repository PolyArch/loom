add_library(LoomMapping
  ConfiguredBoundaryProjection.cpp
  FabricOccurrenceIndex.cpp
  FabricRoutingIndex.cpp
  PairedLane.cpp
  Verifier.cpp
)

target_include_directories(LoomMapping PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomMapping PUBLIC
  LoomCommon
  LoomFabricBoundaryDataPath
  LLVMSupport
)
