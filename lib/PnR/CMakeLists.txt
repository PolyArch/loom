add_mlir_library(LoomPnR
  EndpointRouter.cpp
  FrozenConstraintIndex.cpp
  HandshakeCandidateState.cpp
  InitializerRelationSolver.cpp
  IncrementalTopologicalOrder.cpp
  MappingObjective.cpp
  PnrIndex.cpp
  PnrConfig.cpp
  RouteTreeState.cpp
  RoutingNegotiation.cpp
  SpatialBindingRelationModel.cpp
  SpatialCanonicalSeed.cpp
  SpatialCandidateInitializer.cpp
  SpatialCandidateMemory.cpp
  SpatialCandidateResourceTime.cpp
  SpatialCandidateRelations.cpp
  SpatialCandidateRouteResources.cpp
  SpatialCandidateState.cpp
  SpatialMappingMaterializer.cpp
  SpatialNetRouter.cpp
  SpatialObjective.cpp
  SpatialPnrCapacityIndex.cpp
  SpatialPathFinderRouter.cpp
  SpatialPnrHandshakeIndex.cpp
  SpatialPnrMemoryIndex.cpp
  SpatialPnrPortIndex.cpp
  SpatialPnrProblem.cpp
  SpatialPnrResourceIndex.cpp
  SpatialPnrTagContinuityIndex.cpp
  SpatialPnrTransferIndex.cpp
  SpatialRouteCostState.cpp
  SpatialRouteResourceState.cpp
  SpatialTagContinuity.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/PnR

  LINK_LIBS PUBLIC
  LoomFabricIdentity
  LoomFabricTemporalOperandBuffer
  LoomDSEObjective
  LoomMappingArtifact
  LLVMSupport
)

target_include_directories(LoomPnR PRIVATE
  ${LOOM_SOURCE_DIR}/lib
)
