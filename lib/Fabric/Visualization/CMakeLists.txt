add_mlir_library(LoomFabricVisualization
  FabricLayeredLayout.cpp
  FabricVisualizationHtml.cpp
  FabricVisualizationModel.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric/Visualization

)

target_link_libraries(LoomFabricVisualization PUBLIC
  LLVM
  LoomCommon
  LoomFabricArtifact
  MLIRFabric
)
mlir_target_link_libraries(LoomFabricVisualization PUBLIC
  MLIRIR
)
