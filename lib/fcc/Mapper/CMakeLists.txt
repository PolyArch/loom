add_library(FccMapper STATIC
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

target_include_directories(FccMapper PUBLIC
  ${FCC_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(FccMapper PUBLIC
  FccFabric
  CIRCTHandshake
  MLIRIR
  MLIRSupport
)

if(FCC_ENABLE_ORTOOLS AND TARGET ortools::ortools)
  target_link_libraries(FccMapper PUBLIC ortools::ortools)
  target_compile_definitions(FccMapper PRIVATE FCC_HAVE_ORTOOLS=1)
  target_compile_features(FccMapper PRIVATE cxx_std_20)
endif()
