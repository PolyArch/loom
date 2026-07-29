add_library(LoomStructuredFabricAnalyticModel
  AnalyticModelSupport.cpp
  CanonicalDataflowFabricAnalytic.cpp
  StructuredFabricAnalytic.cpp
  StructuredProgramFunctional.cpp
)

target_include_directories(LoomStructuredFabricAnalyticModel PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomStructuredFabricAnalyticModel PUBLIC
  LoomEvaluation
  LoomFabricArtifact
  LoomFrontendCompilation
  LoomNativeSimulationOracle
  LoomSimulator
  MLIRDataflow
  MLIRLoom
  MLIRIR
)
