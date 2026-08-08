add_executable(loom-pnr-index-test
  PnrIndexTest.cpp
)

add_executable(loom-system-pnr-search-domain-test
  SystemPnrSearchDomainTest.cpp
)

add_executable(loom-system-pnr-service-domain-test
  SystemPnrServiceDomainTest.cpp
)

add_executable(loom-system-candidate-state-test
  SystemCandidateStateTest.cpp
  SystemCandidateStateTestSupport.cpp
)

add_executable(loom-system-channel-applicability-test
  SystemChannelApplicabilityTest.cpp
)

target_link_libraries(loom-system-pnr-search-domain-test PRIVATE
  LoomADGBuilder
  LoomPnR
  MLIRArithDialect
  MLIRDataflow
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSCFDialect
  LLVMSupport
)

target_link_libraries(loom-system-pnr-service-domain-test PRIVATE
  LoomADGBuilder
  LoomPnR
  MLIRArithDialect
  MLIRDataflow
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  LLVMSupport
)

target_link_libraries(loom-system-candidate-state-test PRIVATE
  LoomADGBuilder
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

target_link_libraries(loom-system-channel-applicability-test PRIVATE
  LoomADGBuilder
  LoomDSE
  LoomDSEMappingGenerator
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

target_link_libraries(loom-pnr-index-test PRIVATE
  LoomPnR
  LLVMSupport
)

add_executable(loom-pnr-routing-negotiation-test
  RoutingNegotiationTest.cpp
)

target_link_libraries(loom-pnr-routing-negotiation-test PRIVATE
  LoomPnR
  LLVMSupport
)

add_executable(loom-pnr-config-test
  PnrConfigTest.cpp
)

target_link_libraries(loom-pnr-config-test PRIVATE
  LoomPnR
  LLVMSupport
)

add_executable(loom-pnr-incremental-topology-test
  IncrementalTopologicalOrderTest.cpp
)

target_include_directories(loom-pnr-incremental-topology-test PRIVATE
  ${LOOM_SOURCE_DIR}/lib/PnR
)

target_link_libraries(loom-pnr-incremental-topology-test PRIVATE
  LoomPnR
  LLVMSupport
)

add_executable(loom-pnr-endpoint-router-test
  EndpointRouterTest.cpp
)

target_link_libraries(loom-pnr-endpoint-router-test PRIVATE
  LoomPnR
  LLVMSupport
)

add_executable(loom-pnr-initializer-relation-solver-test
  InitializerRelationSolverTest.cpp
)

target_include_directories(loom-pnr-initializer-relation-solver-test PRIVATE
  ${LOOM_SOURCE_DIR}/lib/PnR
)

target_link_libraries(loom-pnr-initializer-relation-solver-test PRIVATE
  LoomPnR
  LLVMSupport
)

add_executable(loom-pnr-deterministic-search-protocol-test
  DeterministicSearchProtocolTest.cpp
)

target_link_libraries(loom-pnr-deterministic-search-protocol-test PRIVATE
  LoomPnR
  LLVMSupport
)

add_executable(loom-pnr-spatial-action-proposal-test
  SpatialActionProposalTest.cpp
)

target_link_libraries(loom-pnr-spatial-action-proposal-test PRIVATE
  LoomPnR
  LLVMSupport
)

add_executable(loom-pnr-cp-sat-exact-protocol-test
  CpSatExactProtocolTest.cpp
)

target_include_directories(loom-pnr-cp-sat-exact-protocol-test PRIVATE
  ${LOOM_SOURCE_DIR}/lib/PnR/CpSat
)

target_link_libraries(loom-pnr-cp-sat-exact-protocol-test PRIVATE
  LoomPnRCpSat
  LLVMSupport
)
