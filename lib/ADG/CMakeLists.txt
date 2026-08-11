add_mlir_library(LoomADGBuilder
  Builtin.cpp
  Builder.cpp
  BuilderDerivation.cpp
  SystemBuilderDerivation.cpp
  BuilderDomainAuthoring.cpp
  Export.cpp
  FuLibrary.cpp
  MemoryLibrary.cpp
  MeshSwitchNetwork.cpp
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
  LoomFabricSwitchResourceContract
  MLIRFabricEnums
  MLIRFabric
)
mlir_target_link_libraries(LoomADGBuilder PUBLIC
  MLIRIR
)
