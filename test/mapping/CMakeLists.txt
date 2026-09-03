add_executable(loom-mapping-artifact-test
  MappingArtifactTest.cpp
)

add_executable(loom-system-mapping-constraint-set-test
  SystemMappingConstraintSetTest.cpp
)

add_executable(loom-system-service-obligation-test
  SystemServiceObligationTest.cpp
)

add_executable(loom-system-service-route-wire-test
  SystemServiceRouteWireTest.cpp
)

add_library(loom-tech-mapping-artifact-test-support STATIC
  ../TestAllocationProbe.cpp
  HandshakeProjectionTestSupport.cpp
  SpatialMemoryActionTestSupport.cpp
  SpatialProgressWitnessTestSupport.cpp
  TemporalComputeFabricTestSupport.cpp
  TechMappingArtifactTest.cpp
  TechMappingCandidateTestSupport.cpp
  SpatialRouteConstraintTestSupport.cpp
)

target_include_directories(loom-tech-mapping-artifact-test-support PRIVATE
  ${LOOM_SOURCE_DIR}/lib/Mapping/Artifact
  ${LOOM_SOURCE_DIR}/lib/PnR
)

add_executable(loom-tech-mapping-artifact-test
  TechMappingArtifactTestMain.cpp
)

add_executable(loom-tech-mapping-compute-boundary-test
  TechMappingComputeBoundaryTest.cpp
)

add_executable(loom-mapping-progress-test
  MappingProgressTest.cpp
)

add_executable(loom-spatial-candidate-workflow-test
  SpatialCandidateWorkflowTest.cpp
)

add_executable(loom-tech-mapping-config-test
  TechMappingConfigTest.cpp
)

add_executable(loom-tech-mapping-generator-test
  ../TestAllocationProbe.cpp
  ../InternalMemoryEdgeTestSupport.cpp
  TechMappingGeneratorTest.cpp
  SpatialMemoryConstraintTestSupport.cpp
)

target_include_directories(loom-tech-mapping-generator-test PRIVATE
  ${LOOM_SOURCE_DIR}/lib/Mapping/Tech
)

add_executable(loom-tech-mapping-activity-definedness-test
  TechMappingActivityDefinednessTest.cpp
)

target_include_directories(loom-tech-mapping-activity-definedness-test PRIVATE
  ${LOOM_SOURCE_DIR}/lib/Mapping/Tech
)

add_executable(loom-tech-mapping-search-test
  TechMappingSearchTest.cpp
)

add_executable(loom-tech-mapping-joint-legality-test
  TechMappingJointLegalityTest.cpp
)

add_executable(loom-spatial-mapping-wire-test
  SpatialMappingWireTest.cpp
)

add_executable(loom-spatial-mapping-artifact-test
  ../TestAllocationProbe.cpp
  CgraAdmissionTestSupport.cpp
  HandshakeProjectionTestSupport.cpp
  SpatialCandidateSelectionTestSupport.cpp
  SpatialMemoryActionTestSupport.cpp
  SpatialMappingArtifactTest.cpp
  SpatialMemoryMappingArtifactTestSupport.cpp
  SpatialRuntimeCounterexampleExactRepairTestSupport.cpp
  SpatialRuntimeCounterexampleTestSupport.cpp
  TemporalMappingFabricTestSupport.cpp
  TemporalComputeFabricTestSupport.cpp
  TemporalPeTagDomainTestSupport.cpp
  TechMappingCandidateTestSupport.cpp
)

add_executable(loom-spatial-memory-service-handshake-test
  ../TestAllocationProbe.cpp
  CgraAdmissionTestSupport.cpp
  HandshakeProjectionTestSupport.cpp
  SpatialCandidateSelectionTestSupport.cpp
  SpatialMemoryActionTestSupport.cpp
  SpatialMemoryMappingArtifactTestSupport.cpp
  SpatialMemoryServiceHandshakeTest.cpp
  TechMappingCandidateTestSupport.cpp
)

add_executable(loom-spatial-tag-continuity-index-test
  SpatialTagContinuityIndexTest.cpp
)

loom_enable_allocation_probe(loom-tech-mapping-artifact-test)
loom_enable_allocation_probe(loom-tech-mapping-compute-boundary-test)
loom_enable_allocation_probe(loom-spatial-candidate-workflow-test)
loom_enable_allocation_probe(loom-tech-mapping-generator-test)
loom_enable_allocation_probe(loom-spatial-mapping-artifact-test)
loom_enable_allocation_probe(loom-spatial-memory-service-handshake-test)

add_executable(loom-configured-hardware-projection-test
  ConfiguredHardwareProjectionTest.cpp
)

target_include_directories(loom-configured-hardware-projection-test PRIVATE
  ${LOOM_SOURCE_DIR}/lib/Mapping/Artifact
  ${LOOM_SOURCE_DIR}/test/hardware
)

target_include_directories(loom-spatial-mapping-artifact-test PRIVATE
  ${LOOM_SOURCE_DIR}/lib/Application
  ${LOOM_SOURCE_DIR}/lib/Mapping/Artifact
  ${LOOM_SOURCE_DIR}/lib/PnR
  ${LOOM_SOURCE_DIR}/lib/Simulator
)

target_include_directories(loom-spatial-memory-service-handshake-test PRIVATE
  ${LOOM_SOURCE_DIR}/lib/Application
  ${LOOM_SOURCE_DIR}/lib/Mapping/Artifact
  ${LOOM_SOURCE_DIR}/lib/PnR
  ${LOOM_SOURCE_DIR}/lib/Simulator
)

target_link_libraries(loom-mapping-artifact-test PRIVATE
  LoomMappingArtifact
  MLIRIR
  MLIRParser
  LLVMSupport
)

target_link_libraries(loom-system-mapping-constraint-set-test PRIVATE
  LoomADGBuilder
  LoomMappingArtifact
  LoomPnR
  LoomPnRGenerator
  LoomTechMapping
  MLIRArithDialect
  MLIRDataflow
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  LLVMSupport
)

target_link_libraries(loom-system-service-obligation-test PRIVATE
  LoomMappingArtifact
  MLIRArithDialect
  MLIRDataflow
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  LLVMSupport
)

target_link_libraries(loom-system-service-route-wire-test PRIVATE
  LoomMappingArtifact
  MLIRIR
  MLIRParser
  LLVMSupport
)

target_link_libraries(loom-tech-mapping-artifact-test-support PUBLIC
  LoomADGBuilder
  LoomMappingArtifact
  LoomPnR
  LoomPnRCpSat
  MLIRArithDialect
  MLIRDataflow
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRParser
  LLVMSupport
)

target_link_libraries(loom-tech-mapping-artifact-test PRIVATE
  loom-tech-mapping-artifact-test-support
)

target_link_libraries(loom-tech-mapping-compute-boundary-test PRIVATE
  loom-tech-mapping-artifact-test-support
)

target_link_libraries(loom-mapping-progress-test PRIVATE
  LoomMappingArtifact
  MLIRDataflow
  MLIRIR
  MLIRParser
  LLVMSupport
)

target_link_libraries(loom-spatial-candidate-workflow-test PRIVATE
  loom-tech-mapping-artifact-test-support
)

target_link_libraries(loom-tech-mapping-config-test PRIVATE
  LoomCommon
  LoomTechMapping
  LLVMSupport
)

target_link_libraries(loom-tech-mapping-generator-test PRIVATE
  LoomADGBuilder
  LoomMappingArtifact
  LoomPnR
  LoomTechMapping
  MLIRArithDialect
  MLIRDataflow
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRParser
  LLVMSupport
)

target_link_libraries(loom-tech-mapping-activity-definedness-test PRIVATE
  LoomADGBuilder
  LoomMappingArtifact
  LoomTechMapping
  MLIRArithDialect
  MLIRDataflow
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMapping
  MLIRParser
  MLIRSupport
  LLVMSupport
)

target_include_directories(loom-tech-mapping-search-test PRIVATE
  ${LOOM_SOURCE_DIR}/lib/Mapping/Tech
)

target_link_libraries(loom-tech-mapping-search-test PRIVATE
  LoomCommon
  LoomTechMapping
  LLVMSupport
)

target_link_libraries(loom-tech-mapping-joint-legality-test PRIVATE
  LoomADGBuilder
  LoomMappingArtifact
  LoomPnR
  LoomPnRGenerator
  LoomTechMapping
  MLIRArithDialect
  MLIRDataflow
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRParser
  LLVMSupport
)

target_link_libraries(loom-spatial-mapping-wire-test PRIVATE
  LoomMappingArtifact
  MLIRIR
  MLIRParser
  LLVMSupport
)

target_link_libraries(loom-spatial-mapping-artifact-test PRIVATE
  LoomADGBuilder
  LoomApplicationRuntimeManifest
  LoomDSEMappingGenerator
  LoomMappingArtifact
  LoomMappingInspection
  LoomStructuredFabricAnalyticModel
  LoomPnR
  LoomPnRCpSat
  LoomPnRGenerator
  LoomTechMapping
  LoomSimulator
  MLIRArithDialect
  MLIRDataflow
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMapping
  MLIRMemRefDialect
  MLIRParser
  LLVMSupport
)

target_link_libraries(loom-spatial-memory-service-handshake-test PRIVATE
  LoomADGBuilder
  LoomApplicationRuntimeManifest
  LoomDSEMappingGenerator
  LoomMappingArtifact
  LoomMappingInspection
  LoomStructuredFabricAnalyticModel
  LoomPnR
  LoomPnRCpSat
  LoomPnRGenerator
  LoomTechMapping
  LoomSimulator
  MLIRArithDialect
  MLIRDataflow
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMapping
  MLIRMemRefDialect
  MLIRParser
  LLVMSupport
)

target_link_libraries(loom-spatial-tag-continuity-index-test PRIVATE
  LoomFabricArtifact
  LoomPnR
  MLIRIR
  MLIRParser
  LLVMSupport
)

target_link_libraries(loom-configured-hardware-projection-test PRIVATE
  LoomADGBuilder
  LoomDeployment
  LoomHardwareConfigurationTestSupport
  LoomMappingArtifact
  LoomPnR
  LoomPnRCpSat
  LoomPnRGenerator
  LoomSimulator
  LoomTechMapping
  MLIRArithDialect
  MLIRDataflow
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMapping
  MLIRParser
  MLIRVectorDialect
  LLVMSupport
)
