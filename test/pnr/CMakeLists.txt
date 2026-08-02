add_executable(loom-pnr-index-test
  PnrIndexTest.cpp
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
