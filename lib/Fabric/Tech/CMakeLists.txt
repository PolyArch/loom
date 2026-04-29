add_mlir_library(MLIRFabricTech
  EnumerateFuSubgraphsPass.cpp
  MapSubgraphToFusPass.cpp
  PartitionGraphPass.cpp
  SubgraphEnumerator.cpp
  SubgraphGraphView.cpp
  SubgraphMatcher.cpp
  TemplateLibrary.cpp
  Partitioner/BeamPartitioner.cpp
  Partitioner/CandidateCache.cpp
  Partitioner/CostModel.cpp
  Partitioner/GreedyPartitioner.cpp
  Partitioner/ILPPartitioner.cpp
  Partitioner/ListPartitioner.cpp
  Partitioner/Materializer.cpp
  Partitioner/Partitioner.cpp
  Partitioner/PartitionerCommon.cpp
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

if(LOOM_ENABLE_ILP)
  target_link_libraries(MLIRFabricTech PUBLIC highs::highs)
  set_property(SOURCE Partitioner/ILPPartitioner.cpp APPEND PROPERTY
                COMPILE_OPTIONS -frtti -fexceptions)
endif()

add_subdirectory(Synthesizer)
