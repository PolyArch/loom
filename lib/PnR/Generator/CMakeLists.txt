add_mlir_library(LoomPnRGenerator
  FabricTopologyQualityDiagnostic.cpp
  SpatialPnrGenerator.cpp
  SystemPnrGenerator.cpp

  LINK_LIBS PUBLIC
  LoomPnR
  LoomPnRCpSat
  LoomFabricArtifact
  LLVMSupport
)

target_include_directories(LoomPnRGenerator PRIVATE
  ${LOOM_SOURCE_DIR}/lib/PnR
)
