add_executable(loom-fabric-elaboration-api-test
  ElaborationApiTest.cpp
)

add_executable(loom-fabric-elaboration-alias-chain-test
  ElaborationAliasChainTest.cpp
)

add_executable(loom-fabric-elaboration-header-test
  ElaborationHeaderTest.cpp
)

add_executable(loom-fabric-boundary-data-path-test
  BoundaryDataPathTest.cpp
)

add_executable(loom-fabric-resource-contract-test
  ResourceContractTest.cpp
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

target_link_libraries(loom-fabric-boundary-data-path-test PRIVATE
  MLIRFabric
)

target_link_libraries(loom-fabric-resource-contract-test PRIVATE
  LoomFabricResourceContract
)
