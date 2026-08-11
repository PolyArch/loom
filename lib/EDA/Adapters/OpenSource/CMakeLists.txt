add_library(LoomEDAOpenSource
  MappedRtlSimulation.cpp
  MappedRtlSimulationFacts.cpp
  MappedRtlSimulationHarness.cpp
  MappedRtlSimulationResult.cpp
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
  LoomStructuredFabricAnalyticModel
  LLVMSupport
)
