add_library(FccMapper STATIC
  Graph.cpp
  BridgeBinding.cpp
  TagRuntime.cpp
  ADGFlattener.cpp
  DFGBuilder.cpp
  MappingState.cpp
  TechMapper.cpp
  Mapper.cpp
  MapperRouting.cpp
  ConfigGen.cpp
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
