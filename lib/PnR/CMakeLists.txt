add_mlir_library(LoomPnR
  FrozenConstraintIndex.cpp
  PnrIndex.cpp
  PnrConfig.cpp
  RouteTreeState.cpp
  RoutingNegotiation.cpp
  SpatialPnrHandshakeIndex.cpp
  SpatialPnrPortIndex.cpp
  SpatialPnrProblem.cpp
  SpatialPnrResourceIndex.cpp
  SpatialPnrTransferIndex.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/PnR

  LINK_LIBS PUBLIC
  LoomFabricIdentity
  LoomMappingArtifact
  LLVMSupport
)

target_include_directories(LoomPnR PRIVATE
  ${LOOM_SOURCE_DIR}/lib
)
