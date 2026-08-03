add_mlir_library(LoomPnR
  EndpointRouter.cpp
  FrozenConstraintIndex.cpp
  HandshakeCandidateState.cpp
  IncrementalTopologicalOrder.cpp
  MappingObjective.cpp
  PnrIndex.cpp
  PnrConfig.cpp
  RouteTreeState.cpp
  RoutingNegotiation.cpp
  SpatialCandidateRouteResources.cpp
  SpatialCandidateState.cpp
  SpatialNetRouter.cpp
  SpatialPathFinderRouter.cpp
  SpatialPnrHandshakeIndex.cpp
  SpatialPnrPortIndex.cpp
  SpatialPnrProblem.cpp
  SpatialPnrResourceIndex.cpp
  SpatialPnrTransferIndex.cpp
  SpatialRouteCostState.cpp
  SpatialRouteResourceState.cpp

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
