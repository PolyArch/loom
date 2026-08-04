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
