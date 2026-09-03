add_executable(loom-mapped-rtl-hierarchy-verilator
  MappedRtlHierarchyVerilatorLauncher.cpp
)

target_include_directories(loom-mapped-rtl-hierarchy-verilator PRIVATE
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(loom-mapped-rtl-hierarchy-verilator PRIVATE
  LoomCommon
  LLVMSupport
)

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

# The launcher is frozen into every mapped-RTL bundle as an auxiliary tool, so
# any consumer of the adapters must have it built.
add_dependencies(LoomMappedRtlExecution loom-mapped-rtl-hierarchy-verilator)

target_compile_definitions(LoomMappedRtlExecution PRIVATE
  LOOM_MAPPED_RTL_HIERARCHY_LAUNCHER_PATH="$<TARGET_FILE:loom-mapped-rtl-hierarchy-verilator>"
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
