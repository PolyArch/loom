add_library(FccMapper STATIC
  Graph.cpp
  BridgeBinding.cpp
  TagRuntime.cpp
  ADGFlattener.cpp
  ADGFlattenerWiring.cpp
  DFGBuilder.cpp
  MappingState.cpp
  TechMapper.cpp
  TechMapperVariants.cpp
  Mapper.cpp
  MapperPlacement.cpp
  MapperRefinement.cpp
  MapperRouting.cpp
  MapperRoutingStrategy.cpp
  ConfigGen.cpp
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
