add_library(LoomStructuredFabricAnalyticModel
  AnalyticModelSupport.cpp
  CanonicalDataflowFabricAnalytic.cpp
  CgraSimulation.cpp
  DfgSimulation.cpp
  SimulationComparison.cpp
  StructuredEvaluationInvocationCache.cpp
  StructuredFabricAnalytic.cpp
  StructuredProgramFunctional.cpp
)

target_include_directories(LoomStructuredFabricAnalyticModel PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomStructuredFabricAnalyticModel PUBLIC
  LoomConfig
  LoomEvaluation
  LoomFabricArtifact
  LoomFrontendCompilation
  LoomNativeSimulationOracle
  LoomSimulator
  MLIRDataflow
  MLIRLoom
  MLIRIR
)
