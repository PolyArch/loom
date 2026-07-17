add_library(LoomMapping
  FabricOccurrenceIndex.cpp
  FabricRoutingIndex.cpp
  Verifier.cpp
)

target_include_directories(LoomMapping PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomMapping PUBLIC
  LoomFabricBoundaryDataPath
  LLVMSupport
)
