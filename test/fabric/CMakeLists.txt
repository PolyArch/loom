add_executable(loom-fabric-elaboration-api-test
  ElaborationApiTest.cpp
)

add_executable(loom-fabric-elaboration-alias-chain-test
  ElaborationAliasChainTest.cpp
)

add_executable(loom-fabric-elaboration-header-test
  ElaborationHeaderTest.cpp
)

target_link_libraries(loom-fabric-elaboration-api-test PRIVATE
  MLIRFabric
  MLIRParser
)

target_link_libraries(loom-fabric-elaboration-alias-chain-test PRIVATE
  MLIRFabric
  MLIRParser
)

target_link_libraries(loom-fabric-elaboration-header-test PRIVATE
  MLIRFabric
)
