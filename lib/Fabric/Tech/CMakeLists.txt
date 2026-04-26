add_mlir_library(MLIRFabricTech
  EnumerateFuSubgraphsPass.cpp
  MapSubgraphToFusPass.cpp
  PartitionGraphPass.cpp
  SubgraphEnumerator.cpp
  SubgraphMatcher.cpp
  TemplateLibrary.cpp
  Partitioner/BeamPartitioner.cpp
  Partitioner/CostModel.cpp
  Partitioner/GreedyPartitioner.cpp
  Partitioner/ListPartitioner.cpp
  Partitioner/Materializer.cpp
  Partitioner/Partitioner.cpp
  Partitioner/SAPartitioner.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric/Tech
  ${LOOM_SOURCE_DIR}/include/Fabric/Tech/Partitioner

  LINK_LIBS PUBLIC
  LoomCommon
  MLIRArithDialect
  MLIRDataflow
  MLIRFabric
  MLIRFuncDialect
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRSupport
)
