add_library(LoomHardwareConfigurationTestSupport
  ConfigurationABI2TestSupport.cpp
)

target_include_directories(LoomHardwareConfigurationTestSupport PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(LoomHardwareConfigurationTestSupport PUBLIC
  LoomADGBuilder
  LoomHardwareConfiguration
  MLIRDataflow
  MLIRFabric
  MLIRIR
  LLVMSupport
)

add_executable(loom-hardware-configuration-abi-test
  ConfigurationABITest.cpp
)

target_link_libraries(loom-hardware-configuration-abi-test PRIVATE
  LoomHardwareConfiguration
  MLIRDataflow
  MLIRFabric
  MLIRIR
  MLIRParser
  LLVMSupport
)

add_executable(loom-hardware-implementation-test
  HardwareImplementationTest.cpp
)

target_link_libraries(loom-hardware-implementation-test PRIVATE
  LoomHardwareConfigurationTestSupport
  LoomADGBuilder
  LoomHardwareImplementation
  LoomHardwareConfiguration
  LoomImplementationPlatform
  MLIRDataflow
  MLIRFabric
  MLIRIR
  MLIRParser
  LLVMSupport
)

add_executable(loom-representation-format-test
  RepresentationFormatTest.cpp
)

target_link_libraries(loom-representation-format-test PRIVATE
  LoomHardwareImplementation
  LLVMSupport
)

add_executable(loom-representation-locator-test
  RepresentationLocatorTest.cpp
)

target_link_libraries(loom-representation-locator-test PRIVATE
  LoomHardwareImplementation
  LLVMSupport
)

add_executable(loom-implementation-payload-test
  ImplementationPayloadTest.cpp
)

target_link_libraries(loom-implementation-payload-test PRIVATE
  LoomHardwareImplementation
  LLVMSupport
)

add_executable(loom-implementation-representation-root-test
  ImplementationRepresentationRootTest.cpp
)

target_link_libraries(loom-implementation-representation-root-test PRIVATE
  LoomHardwareImplementation
  LLVMSupport
)

if(LOOM_HAVE_CIRCT)
  add_library(LoomPortableProviderTestSupport
    PortableProviderTestSupport.cpp
  )

  target_include_directories(LoomPortableProviderTestSupport PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
  )

  target_link_libraries(LoomPortableProviderTestSupport PUBLIC
    LoomHardwareRTL
    LLVMSupport
  )

  add_executable(loom-portable-provider-test-support-test
    PortableProviderTestSupportTest.cpp
  )

  target_link_libraries(loom-portable-provider-test-support-test PRIVATE
    LoomPortableProviderTestSupport
    LLVMSupport
  )

  add_executable(loom-representation-index-test
    RepresentationIndexTest.cpp
  )

  target_link_libraries(loom-representation-index-test PRIVATE
    LoomHardwareRepresentationIndex
    LLVMSupport
  )

  add_executable(loom-representation-gate-index-test
    RepresentationGateIndexTest.cpp
  )

  target_link_libraries(loom-representation-gate-index-test PRIVATE
    LoomHardwareRepresentationIndex
    LLVMSupport
  )

  add_executable(loom-common-skeleton-test
    CommonSkeletonTest.cpp
  )

  target_link_libraries(loom-common-skeleton-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomADGBuilder
    LoomHardwareRTL
    MLIRFabric
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-specialization-test
    SpecializationTest.cpp
  )

  target_link_libraries(loom-specialization-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomHardwareConfiguration
    LoomHardwareRTL
    LoomImplementationPlatform
    LLVMSupport
  )

  add_executable(loom-portable-provider-coverage-test
    PortableProviderCoverageTest.cpp
  )

  target_link_libraries(loom-portable-provider-coverage-test PRIVATE
    LoomHardwareRTL
    LLVMSupport
  )

  add_executable(loom-portable-scalar-integer-add-sub-test
    PortableScalarIntegerAddSubTest.cpp
  )

  target_link_libraries(loom-portable-scalar-integer-add-sub-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-scalar-integer-cast-test
    PortableScalarIntegerCastTest.cpp
  )

  target_link_libraries(loom-portable-scalar-integer-cast-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-integer-shift-test
    PortableIntegerShiftTest.cpp
  )

  target_link_libraries(loom-portable-integer-shift-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-integer-count-zeros-test
    PortableIntegerCountZerosTest.cpp
  )

  target_link_libraries(loom-portable-integer-count-zeros-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-integer-saturating-add-sub-test
    PortableIntegerSaturatingAddSubTest.cpp
  )

  target_link_libraries(loom-portable-integer-saturating-add-sub-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-scalar-integer-compare-min-max-test
    PortableScalarIntegerCompareMinMaxTest.cpp
  )

  target_link_libraries(loom-portable-scalar-integer-compare-min-max-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-scalar-integer-multiply-test
    PortableScalarIntegerMultiplyTest.cpp
  )

  target_link_libraries(loom-portable-scalar-integer-multiply-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-scalar-unsigned-integer-div-rem-test
    PortableScalarUnsignedIntegerDivRemTest.cpp
  )

  target_link_libraries(loom-portable-scalar-unsigned-integer-div-rem-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-scalar-signed-integer-div-rem-test
    PortableScalarSignedIntegerDivRemTest.cpp
  )

  target_link_libraries(loom-portable-scalar-signed-integer-div-rem-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-fixed-vector-integer-add-sub-test
    PortableFixedVectorIntegerAddSubTest.cpp
  )

  target_link_libraries(loom-portable-fixed-vector-integer-add-sub-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-fixed-vector-integer-compare-min-max-test
    PortableFixedVectorIntegerCompareMinMaxTest.cpp
  )

  target_link_libraries(loom-portable-fixed-vector-integer-compare-min-max-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-fixed-vector-integer-multiply-test
    PortableFixedVectorIntegerMultiplyTest.cpp
  )

  target_link_libraries(loom-portable-fixed-vector-integer-multiply-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-fixed-vector-value-select-test
    PortableFixedVectorValueSelectTest.cpp
  )

  target_link_libraries(loom-portable-fixed-vector-value-select-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomADGBuilder
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-integer-logic-test
    PortableIntegerLogicTest.cpp
  )

  target_link_libraries(loom-portable-integer-logic-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-fixed-vector-pack-unpack-test
    PortableFixedVectorPackUnpackTest.cpp
  )

  target_link_libraries(loom-portable-fixed-vector-pack-unpack-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-fixed-vector-parallelize-serialize-test
    PortableFixedVectorParallelizeSerializeTest.cpp
  )

  target_link_libraries(loom-portable-fixed-vector-parallelize-serialize-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomADGBuilder
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-fixed-vector-shuffle-test
    PortableFixedVectorShuffleTest.cpp
  )

  target_link_libraries(loom-portable-fixed-vector-shuffle-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomADGBuilder
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-fixed-vector-slice-align-merge-test
    PortableFixedVectorSliceAlignMergeTest.cpp
  )

  target_link_libraries(loom-portable-fixed-vector-slice-align-merge-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomADGBuilder
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-loop-carry-test
    PortableLoopCarryTest.cpp
  )

  target_link_libraries(loom-portable-loop-carry-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-loop-gate-test
    PortableLoopGateTest.cpp
  )

  target_link_libraries(loom-portable-loop-gate-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-loop-invariant-test
    PortableLoopInvariantTest.cpp
  )

  target_link_libraries(loom-portable-loop-invariant-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-loop-stream-test
    PortableLoopStreamTest.cpp
  )

  target_link_libraries(loom-portable-loop-stream-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomADGBuilder
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-token-constant-sync-test
    PortableTokenConstantSyncTest.cpp
  )

  target_link_libraries(loom-portable-token-constant-sync-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomADGBuilder
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-token-mux-demux-test
    PortableTokenMuxDemuxTest.cpp
  )

  target_link_libraries(loom-portable-token-mux-demux-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomADGBuilder
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-operation-leaf-state-layout-test
    OperationLeafStateLayoutTest.cpp
  )

  target_link_libraries(loom-operation-leaf-state-layout-test PRIVATE
    LoomADGBuilder
    LoomHardwareRTL
    MLIRIR
    LLVMSupport
  )

  add_executable(loom-control-stream-structure-test
    ControlStreamStructureTest.cpp
  )

  target_link_libraries(loom-control-stream-structure-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomADGBuilder
    LoomHardwareRTL
    MLIRIR
    LLVMSupport
  )

  add_executable(loom-portable-scalar-bit-reinterpret-test
    PortableScalarBitReinterpretTest.cpp
  )

  target_link_libraries(loom-portable-scalar-bit-reinterpret-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-scalar-float-fma-test
    PortableScalarFloatFmaTest.cpp
  )

  target_link_libraries(loom-portable-scalar-float-fma-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-fixed-vector-float-fma-test
    PortableFixedVectorFloatFmaTest.cpp
  )

  target_link_libraries(loom-portable-fixed-vector-float-fma-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-float-sign-test
    PortableFloatSignTest.cpp
  )

  target_link_libraries(loom-portable-float-sign-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-float-add-sub-test
    PortableFloatAddSubTest.cpp
  )

  target_link_libraries(loom-portable-float-add-sub-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-float-compare-min-max-test
    PortableFloatCompareMinMaxTest.cpp
  )

  target_link_libraries(loom-portable-float-compare-min-max-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomHardwareRTL
    LoomSimulator
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-float-conversions-test
    PortableFloatConversionsTest.cpp
  )

  target_link_libraries(loom-portable-float-conversions-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomHardwareRTL
    LoomSimulator
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-float-multiply-test
    PortableFloatMultiplyTest.cpp
  )

  target_link_libraries(loom-portable-float-multiply-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-float-divide-remainder-test
    PortableFloatDivideRemainderTest.cpp
  )

  target_link_libraries(loom-portable-float-divide-remainder-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-math-root-test
    PortableMathRootTest.cpp
  )

  target_link_libraries(loom-portable-math-root-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomHardwareRTL
    LoomSimulator
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-math-rounding-test
    PortableMathRoundingTest.cpp
  )

  target_link_libraries(loom-portable-math-rounding-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomHardwareRTL
    LoomSimulator
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-portable-scalar-value-select-test
    PortableScalarValueSelectTest.cpp
  )

  target_link_libraries(loom-portable-scalar-value-select-test PRIVATE
    LoomHardwareConfigurationTestSupport
    LoomPortableProviderTestSupport
    LoomADGBuilder
    LoomHardwareRTL
    MLIRDataflow
    MLIRFabric
    MLIRIR
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-circt-conformance-test
    CirctConformanceTest.cpp
  )

  target_link_libraries(loom-circt-conformance-test PRIVATE
    LoomHardwareRTL
    LLVMSupport
  )

  add_executable(loom-hardware-transport-test
    TransportTest.cpp
  )

  target_link_libraries(loom-hardware-transport-test PRIVATE
    LoomFabricArtifact
    LoomHardwareRTL
    MLIRFabric
    MLIRParser
    LLVMSupport
  )

  add_executable(loom-module-boundary-ports-test
    ModuleBoundaryPortsTest.cpp
  )

  target_link_libraries(loom-module-boundary-ports-test PRIVATE
    LoomADGBuilder
    LoomHardwareRTL
    LLVMSupport
  )
endif()

get_property(LOOM_HARDWARE_TEST_TARGETS
  DIRECTORY
  PROPERTY BUILDSYSTEM_TARGETS
)
set(LOOM_HARDWARE_TEST_TARGETS
  ${LOOM_HARDWARE_TEST_TARGETS}
  PARENT_SCOPE
)
