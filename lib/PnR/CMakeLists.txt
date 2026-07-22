add_mlir_library(LoomPnR
  EndpointMatching.cpp
  FrozenComputeDomains.cpp
  FrozenMemoryDomains.cpp
  FrozenRealizationGraph.cpp
  FrozenRoutingGraph.cpp
  PnrIndex.cpp
  PnrProblemInputs.cpp
  RouteTreeState.cpp
  RoutingNegotiation.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/PnR

  LINK_LIBS PUBLIC
  LoomMapping
  LLVMSupport
)

target_include_directories(LoomPnR PRIVATE
  ${LOOM_SOURCE_DIR}/lib
)
