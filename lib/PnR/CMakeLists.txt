add_mlir_library(LoomPnR
  FrozenComputeDomains.cpp
  FrozenRealizationGraph.cpp
  FrozenRoutingGraph.cpp
  PnrIndex.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/PnR

  LINK_LIBS PUBLIC
  LoomMapping
  LLVMSupport
)

target_include_directories(LoomPnR PRIVATE
  ${LOOM_SOURCE_DIR}/lib
)
