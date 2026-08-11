add_library(LoomStructuredFabricAnalyticModel
  AnalyticModelSupport.cpp
  CalibratedFpa.cpp
  CanonicalDataflowFabricAnalytic.cpp
  CanonicalDataflowFunctional.cpp
  CgraSimulation.cpp
  DeterministicGbdt.cpp
  DfgSimulation.cpp
  FixedTabularGbdt.cpp
  FpaParameterContract.cpp
  FabricLowConfidence.cpp
  Gem5SystemCgra.cpp
  Gem5SystemDfg.cpp
  Gem5SystemRtl.cpp
  HardwareImplementationPhysicalCase.cpp
  MappedRtlSimulation.cpp
  OpenRoadStaticFpa.cpp
  ProductionRegistry.cpp
  PredictionCalibration.cpp
  SimulationComparison.cpp
  StructuredEvaluationInvocationCache.cpp
  StructuredFabricAnalytic.cpp
  StructuredProgramFunctional.cpp
  SystemRuntimeParameterContract.cpp
  SystemRuntimePredictor.cpp
)

target_include_directories(LoomStructuredFabricAnalyticModel PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomStructuredFabricAnalyticModel PUBLIC
  LoomConfig
  LoomEDAAsicContracts
  LoomEvaluation
  LoomFabricArtifact
  LoomFrontendCompilation
  LoomGem5SystemExecution
  LoomNativeSimulationOracle
  LoomSimulator
  MLIRDataflow
  MLIRLoom
  MLIRIR
)
