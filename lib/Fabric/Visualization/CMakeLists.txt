add_mlir_library(LoomFabricVisualization
  FabricLayeredLayout.cpp
  FabricVisualizationHtml.cpp
  FabricVisualizationModel.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric/Visualization

  LINK_LIBS PUBLIC
  LoomCommon
  LoomFabricArtifact
  MLIRFabric
  MLIRIR
  LLVMSupport
)
