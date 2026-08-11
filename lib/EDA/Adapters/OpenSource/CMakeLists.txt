add_library(LoomMappedRtlExecution
  MappedRtlExecution.cpp
  MappedRtlSimulationFacts.cpp
  MappedRtlSimulationHarness.cpp
  MappedRtlSimulationResult.cpp
)

target_include_directories(LoomMappedRtlExecution PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomMappedRtlExecution PUBLIC
  LoomDeployment
  LoomEvaluation
  LoomExternalTool
  LoomFabricArtifact
  LoomHardwareConfiguration
  LoomHardwareImplementation
  LoomHardwareRTL
  LoomMappingArtifact
  LoomSimulator
  MLIRDataflow
  LLVMSupport
)

add_library(LoomEDAOpenSource
  MappedRtlSimulation.cpp
  OpenRoad.cpp
  Verilator.cpp
  Yosys.cpp
  YosysGateNetlist.cpp
)

target_include_directories(LoomEDAOpenSource PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomEDAOpenSource PUBLIC
  LoomDSECandidateGenerator
  LoomEDAAsicContracts
  LoomExternalTool
  LoomHardwareImplementation
  LoomHardwareRTL
  LoomImplementationPlatform
  LoomMappedRtlExecution
  LoomStructuredFabricAnalyticModel
  LLVMSupport
)
