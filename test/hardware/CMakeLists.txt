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

if(LOOM_HAVE_CIRCT)
  add_executable(loom-common-skeleton-test
    CommonSkeletonTest.cpp
  )

  target_link_libraries(loom-common-skeleton-test PRIVATE
    LoomADGBuilder
    LoomHardwareRTL
    LLVMSupport
  )

  add_executable(loom-specialization-test
    SpecializationTest.cpp
  )

  target_link_libraries(loom-specialization-test PRIVATE
    LoomADGBuilder
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
    LoomHardwareRTL
    LLVMSupport
  )
endif()
