add_library(LoomMapper STATIC
  MapperOptions.cpp
  Graph.cpp
  BridgeBinding.cpp
  TagRuntime.cpp
  ADGFlattener.cpp
  ADGFlattenerWiring.cpp
  DFGBuilder.cpp
  MappingState.cpp
  OpCompat.cpp
  TechMapper.cpp
  TechMapperSelection.cpp
  TechMapperGraphs.cpp
  TechMapperVariants.cpp
  TopologyModel.cpp
  Mapper.cpp
  MapperSentinels.cpp
  MapperInterleaved.cpp
  MapperPlacement.cpp
  MapperPlacementMethods.cpp
  MapperRefinement.cpp
  MapperLocalRepair.cpp
  MapperLocalRepairExact.cpp
  MapperLocalRepairHotspot.cpp
  MapperLocalRepairMemory.cpp
  MapperLocalRepairEndgame.cpp
  MapperCPSat.cpp
  MapperCongestionEstimator.cpp
  MapperTiming.cpp
  MapperRelaxedRouting.cpp
  MapperRouting.cpp
  MapperRoutingCongestion.cpp
  MapperRoutingStrategy.cpp
  ConfigGen.cpp
  ConfigGenText.cpp
  ConfigGenMapJsonTail.cpp
  ConfigGenReportHelpers.cpp
  ConfigGenHelpers.cpp
  ConfigGenConfig.cpp
)

target_include_directories(LoomMapper PUBLIC
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(LoomMapper PUBLIC
  LoomFabric
  CIRCTHandshake
  MLIRIR
  MLIRSupport
)

if(LOOM_ENABLE_ORTOOLS AND TARGET ortools::ortools)
  target_link_libraries(LoomMapper PUBLIC ortools::ortools)
  target_compile_definitions(LoomMapper PRIVATE LOOM_HAVE_ORTOOLS=1)
  target_compile_features(LoomMapper PRIVATE cxx_std_20)
endif()
