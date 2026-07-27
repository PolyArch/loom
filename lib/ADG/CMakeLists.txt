add_mlir_library(LoomADGBuilder
  Builtin.cpp
  Builder.cpp
  Export.cpp
  FuLibrary.cpp
  MemoryLibrary.cpp
  SystemBuilder.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/ADG

  DEPENDS
  MLIRDataflowEnumsIncGen
  MLIRFabricEnumsIncGen

)

target_link_libraries(LoomADGBuilder PUBLIC
  LLVM
  LoomCommon
  LoomFabricArtifact
  LoomFabricVisualization
  LoomFabricTemporalSwitchResourceContract
  MLIRFabricEnums
  MLIRFabric
)
mlir_target_link_libraries(LoomADGBuilder PUBLIC
  MLIRIR
)
