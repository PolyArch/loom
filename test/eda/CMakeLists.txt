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

add_executable(loom-mapped-rtl-simulation-test
  MappedRtlSimulationTest.cpp
  MappedRtlSimulationTestSupport.cpp
)

target_link_libraries(loom-mapped-rtl-simulation-test PRIVATE
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

add_executable(loom-openroad-placed-test
  OpenRoadPlacedTest.cpp
)

target_link_libraries(loom-openroad-placed-test PRIVATE
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
