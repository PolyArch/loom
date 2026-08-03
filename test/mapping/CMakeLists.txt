add_executable(loom-mapping-artifact-test
  MappingArtifactTest.cpp
)

add_executable(loom-tech-mapping-artifact-test
  TechMappingArtifactTest.cpp
  TechMappingCandidateTestSupport.cpp
)

add_executable(loom-tech-mapping-config-test
  TechMappingConfigTest.cpp
)

add_executable(loom-tech-mapping-generator-test
  TechMappingGeneratorTest.cpp
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
  SpatialMappingArtifactTest.cpp
)

target_link_libraries(loom-mapping-artifact-test PRIVATE
  LoomMappingArtifact
  MLIRIR
  MLIRParser
  LLVMSupport
)

target_link_libraries(loom-tech-mapping-artifact-test PRIVATE
  LoomADGBuilder
  LoomMappingArtifact
  LoomPnR
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

target_link_libraries(loom-tech-mapping-config-test PRIVATE
  LoomCommon
  LoomTechMapping
  LLVMSupport
)

target_link_libraries(loom-tech-mapping-generator-test PRIVATE
  LoomADGBuilder
  LoomMappingArtifact
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
  LoomMappingArtifact
  LoomPnR
  LoomTechMapping
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
