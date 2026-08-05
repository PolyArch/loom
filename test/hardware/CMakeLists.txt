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

if(LOOM_HAVE_CIRCT)
  add_executable(loom-common-skeleton-test
    CommonSkeletonTest.cpp
  )

  target_link_libraries(loom-common-skeleton-test PRIVATE
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
    LoomHardwareConfiguration
    LoomHardwareRTL
    LoomImplementationPlatform
    LLVMSupport
  )

  add_executable(loom-portable-scalar-integer-add-sub-test
    PortableScalarIntegerAddSubTest.cpp
  )

  target_link_libraries(loom-portable-scalar-integer-add-sub-test PRIVATE
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

  add_executable(loom-portable-scalar-bit-reinterpret-test
    PortableScalarBitReinterpretTest.cpp
  )

  target_link_libraries(loom-portable-scalar-bit-reinterpret-test PRIVATE
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
    LoomHardwareRTL
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
