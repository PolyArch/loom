add_library(LoomMappedRtlExecution
  MappedRtlExecution.cpp
  MappedRtlConfigurationTransportReceipt.cpp
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
  MappedRtlRuntimePlatform.cpp
  MappedRtlSimulation.cpp
  OpenRoad.cpp
  OpenRoadRouted.cpp
  OpenRoadStaticFpa.cpp
  Verilator.cpp
  Yosys.cpp
  YosysBlock.cpp
  YosysGateNetlist.cpp
  YosysImplementation.cpp
  YosysSynthesisInvocation.cpp
)

target_include_directories(LoomEDAOpenSource PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomEDAOpenSource PUBLIC
  LoomDSECandidateGenerator
  LoomEDAAsicContracts
  LoomEDAPortableGatePublication
  LoomExternalTool
  LoomHardwareImplementation
  LoomHardwareRTL
  LoomImplementationPlatform
  LoomMappedRtlExecution
  LoomStructuredFabricAnalyticModel
  LLVMSupport
)
