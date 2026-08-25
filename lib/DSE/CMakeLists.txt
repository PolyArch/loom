add_library(LoomDSE
  DataflowEvaluationAcquisition.cpp
  EvidenceObligationSetConfig.cpp
  FpaGroundTruthCollection.cpp
  HardwareImplementationEvaluationAcquisition.cpp
  InvocationManifest.cpp
  JointDesignExploration.cpp
  JointDesignPolicy.cpp
  JointFrontier.cpp
  ResourceTimeFrontier.cpp
  ResourceTimeFrontierSession.cpp
  ResourceTimeFunnel.cpp
  ResourceTimeProjection.cpp
  ResourceTimeSpectrum.cpp
  SpatialMappingEvaluationAcquisition.cpp
  StructuredEvaluationAcquisition.cpp
  StructuredOwnership.cpp
  StructuredOwnershipInvocation.cpp
)

target_include_directories(LoomDSE PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomDSE PUBLIC
  LoomCommon
  LoomConfig
  LoomDSECandidateGenerator
  LoomDSEMappingGenerator
  LoomDSEObjective
  LoomDSEPolicy
  LoomEDAAsicContracts
  LoomEvaluation
  LoomFrontendCompilation
  LoomHardwareImplementation
  LoomMappingArtifact
  LoomNativeSimulationOracle
  LoomPnRGenerator
  LoomSimulator
  LoomStructuredFabricAnalyticModel
  LoomTechMapping
  LLVMSupport
)

add_subdirectory(Generators/Mapping)
add_subdirectory(Generators/Compiler)
add_subdirectory(Generators/Hardware)
add_subdirectory(Generators/Modeling)
add_subdirectory(Feedback)

add_library(LoomDSEProductionOwners
  FuReverseSynthesisWorkflow.cpp
  ProductionOwners.cpp
)

target_include_directories(LoomDSEProductionOwners PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomDSEProductionOwners PUBLIC
  LoomDSE
  LoomDSECompilerGenerator
  LoomDSEFeedback
  LoomDSEHardwareGenerator
  LoomDSEMappingGenerator
  LoomDSEModelingGenerator
  LLVMSupport
)
