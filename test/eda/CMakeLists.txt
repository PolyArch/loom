add_executable(loom-yosys-synthesis-test
  YosysSynthesisTest.cpp
)

target_link_libraries(loom-yosys-synthesis-test PRIVATE
  LoomEDAOpenSource
  LLVMSupport
)

add_executable(loom-yosys-gate-netlist-test
  YosysGateNetlistTest.cpp
)

target_link_libraries(loom-yosys-gate-netlist-test PRIVATE
  LoomHardwareConfigurationTestSupport
  LoomEDAOpenSource
  MLIRParser
  LLVMSupport
)

add_executable(loom-verilator-functional-test
  VerilatorFunctionalTest.cpp
)

target_link_libraries(loom-verilator-functional-test PRIVATE
  LoomEDAOpenSource
  LLVMSupport
)

add_library(LoomMappedRtlSimulationTestSupport STATIC
  MappedRtlSimulationTestSupport.cpp
)

target_include_directories(LoomMappedRtlSimulationTestSupport PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(LoomMappedRtlSimulationTestSupport PUBLIC
  LoomDeploymentTestSupport
  LoomDSE
  LoomDSEMappingGenerator
  LoomEDAOpenSource
  LoomHardwareConfigurationTestSupport
  LoomHardwareRTL
  LoomRootCompleteSpatialPnrTestSupport
  LoomStructuredFabricAnalyticModel
  MLIRArithDialect
  MLIRDLTIDialect
  MLIRDataflow
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  LLVMSupport
)

add_executable(loom-mapped-rtl-simulation-test
  MappedRtlSimulationTest.cpp
)

target_link_libraries(loom-mapped-rtl-simulation-test PRIVATE
  LoomMappedRtlSimulationTestSupport
)

add_executable(loom-mapped-rtl-hierarchy-launcher-test
  MappedRtlHierarchyLauncherTest.cpp
)

target_include_directories(loom-mapped-rtl-hierarchy-launcher-test PRIVATE
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(loom-mapped-rtl-hierarchy-launcher-test PRIVATE
  LLVMSupport
)

target_compile_definitions(loom-mapped-rtl-hierarchy-launcher-test PRIVATE
  LOOM_MAPPED_RTL_HIERARCHY_LAUNCHER_PATH="$<TARGET_FILE:loom-mapped-rtl-hierarchy-verilator>"
)

add_dependencies(loom-mapped-rtl-hierarchy-launcher-test
  loom-mapped-rtl-hierarchy-verilator
)

add_executable(loom-openroad-placed-test
  OpenRoadPlacedTest.cpp
)

target_link_libraries(loom-openroad-placed-test PRIVATE
  LoomHardwareConfigurationTestSupport
  LoomEDAOpenSource
  LLVMSupport
)

add_executable(loom-openroad-routed-test
  OpenRoadRoutedTest.cpp
  OpenRoadPhysicalTestSupport.cpp
)

target_link_libraries(loom-openroad-routed-test PRIVATE
  LoomHardwareConfigurationTestSupport
  LoomEDAOpenSource
  LLVMSupport
)

add_executable(loom-openroad-static-fpa-test
  OpenRoadStaticFpaTest.cpp
  OpenRoadPhysicalTestSupport.cpp
)

target_link_libraries(loom-openroad-static-fpa-test PRIVATE
  LoomConfig
  LoomEvaluation
  LoomHardwareConfigurationTestSupport
  LoomEDAOpenSource
  LLVMSupport
)

add_executable(loom-quartus-prime-test
  QuartusPrimeTest.cpp
)

target_link_libraries(loom-quartus-prime-test PRIVATE
  LoomHardwareConfigurationTestSupport
  LoomEDAIntelAltera
  LLVMSupport
)

add_executable(loom-vivado-static-full-device-test
  VivadoStaticFullDeviceTest.cpp
)

target_link_libraries(loom-vivado-static-full-device-test PRIVATE
  LoomEDAAMD
  LoomHardwareConfigurationTestSupport
  MLIRDataflow
  MLIRFabric
  MLIRIR
  MLIRParser
  LLVMSupport
)

add_executable(loom-cadence-genus-test
  CadenceGenusTest.cpp
)

target_link_libraries(loom-cadence-genus-test PRIVATE
  LoomEDACadence
  LoomHardwareConfigurationTestSupport
  MLIRParser
  LLVMSupport
)

add_executable(loom-synopsys-adapter-test
  SynopsysAdapterTest.cpp
)

target_link_libraries(loom-synopsys-adapter-test PRIVATE
  LoomEDASynopsys
  LLVMSupport
)

add_executable(loom-design-compiler-gate-netlist-test
  DesignCompilerGateNetlistTest.cpp
)

target_link_libraries(loom-design-compiler-gate-netlist-test PRIVATE
  LoomHardwareConfigurationTestSupport
  LoomEDASynopsys
  MLIRParser
  LLVMSupport
)

add_executable(loom-cadence-adapter-test
  CadenceAdapterTest.cpp
)

target_link_libraries(loom-cadence-adapter-test PRIVATE
  LoomEDACadence
  LLVMSupport
)

add_executable(loom-voltus-rail-evaluation-test
  VoltusRailEvaluationTest.cpp
)

target_link_libraries(loom-voltus-rail-evaluation-test PRIVATE
  LoomEDACadence
  LoomHardwareConfigurationTestSupport
  MLIRParser
  LLVMSupport
)

get_property(LOOM_EDA_TEST_TARGETS
  DIRECTORY
  PROPERTY BUILDSYSTEM_TARGETS
)
set(LOOM_EDA_TEST_TARGETS
  ${LOOM_EDA_TEST_TARGETS}
  PARENT_SCOPE
)
