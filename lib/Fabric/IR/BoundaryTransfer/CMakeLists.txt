add_library(LoomFabricBoundaryTransfer
  BoundaryTransfer.cpp
)

target_include_directories(LoomFabricBoundaryTransfer PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomFabricBoundaryTransfer PUBLIC
  LoomFabricResourceContract
)
