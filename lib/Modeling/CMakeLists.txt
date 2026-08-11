add_library(LoomStructuredFabricAnalyticModel
  AnalyticModelSupport.cpp
  CanonicalDataflowFabricAnalytic.cpp
  CanonicalDataflowFunctional.cpp
  CgraSimulation.cpp
  DeterministicGbdt.cpp
  DfgSimulation.cpp
  FixedTabularGbdt.cpp
  FpaParameterContract.cpp
  MappedRtlSimulation.cpp
  OpenRoadStaticFpa.cpp
  ProductionRegistry.cpp
  SimulationComparison.cpp
  StructuredEvaluationInvocationCache.cpp
  StructuredFabricAnalytic.cpp
  StructuredProgramFunctional.cpp
  SystemRuntimeParameterContract.cpp
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
