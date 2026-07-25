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

add_executable(loom-fabric-resource-contract-record-test
  ResourceContractRecordTest.cpp
)

add_executable(loom-fabric-memory-consistency-contract-test
  MemoryConsistencyContractTest.cpp
)

add_executable(loom-fabric-memory-capability-domain-test
  MemoryCapabilityDomainTest.cpp
)

add_executable(loom-fabric-boundary-transfer-test
  BoundaryTransferTest.cpp
)

add_executable(loom-fabric-temporal-operand-buffer-test
  TemporalOperandBufferTest.cpp
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

target_link_libraries(loom-fabric-resource-contract-record-test PRIVATE
  LoomFabricResourceContract
)

target_link_libraries(loom-fabric-memory-consistency-contract-test PRIVATE
  LoomCommon
  LoomFabricIdentity
  LoomFabricMemoryConsistencyContract
  LoomFabricResourceContract
  LLVMSupport
)

target_link_libraries(loom-fabric-memory-capability-domain-test PRIVATE
  LoomCommon
  LoomFabricMemoryCapabilityDomains
  MLIRDLTIDialect
  MLIRDataflow
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
)

target_link_libraries(loom-fabric-boundary-transfer-test PRIVATE
  LoomFabricBoundaryTransfer
)

target_link_libraries(loom-fabric-temporal-operand-buffer-test PRIVATE
  LoomFabricTemporalOperandBuffer
  MLIRFabricEnums
)

add_executable(loom-fabric-implementation-family-test
  ImplementationFamilyTest.cpp
)

target_link_libraries(loom-fabric-implementation-family-test PRIVATE
  LoomFabricImplementationFamily
  MLIRDataflow
  LLVMSupport
)

add_executable(loom-fabric-persistent-ref-test
  PersistentRefTest.cpp
)

add_executable(loom-fabric-artifact-local-reference-test
  ArtifactLocalReferenceTest.cpp
)

add_executable(loom-fabric-fu-capability-template-test
  FuCapabilityTemplateTest.cpp
)

add_executable(loom-fabric-artifact-codec-test
  FabricArtifactCodecTest.cpp
)

add_executable(loom-fabric-artifact-gate-test
  FabricArtifactGateTest.cpp
)

add_executable(loom-fabric-system-contract-test
  FabricSystemContractTest.cpp
)

target_link_libraries(loom-fabric-persistent-ref-test PRIVATE
  LoomCommon
  LoomFabricIdentity
  LLVMSupport
)

target_link_libraries(loom-fabric-artifact-local-reference-test PRIVATE
  LoomCommon
  LoomFabricArtifact
  LoomFabricIdentity
  LLVMSupport
)

target_link_libraries(loom-fabric-fu-capability-template-test PRIVATE
  LoomCommon
  LoomFabricArtifact
  LoomFabricIdentity
  LLVMSupport
)

target_link_libraries(loom-fabric-artifact-codec-test PRIVATE
  LoomCommon
  LoomFabricArtifact
  LLVMSupport
)

target_include_directories(loom-fabric-artifact-gate-test PRIVATE
  ${LOOM_SOURCE_DIR}/lib/Fabric/Artifact
)

target_link_libraries(loom-fabric-artifact-gate-test PRIVATE
  LoomCommon
  LoomFabricArtifact
  LLVMSupport
)

target_link_libraries(loom-fabric-system-contract-test PRIVATE
  LoomCommon
  LoomFabricArtifact
  LoomFabricIdentity
  LLVMSupport
)
