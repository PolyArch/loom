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
