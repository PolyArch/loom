add_library(LoomHardwareRTL
  CirctConformance.cpp
  CommonSkeleton.cpp
  OperationLeaf.cpp
  Providers/FixedVectorIntegerAddSub.cpp
  Providers/FixedVectorPackUnpack.cpp
  Providers/LoopCarry.cpp
  Providers/ProviderSupport.cpp
  Providers/ScalarBitReinterpret.cpp
  Providers/ScalarFloatFma.cpp
  Providers/ScalarIntegerAddSub.cpp
  Providers/ScalarIntegerCompareMinMax.cpp
  Providers/ScalarIntegerMultiply.cpp
  Providers/ScalarValueSelect.cpp
  Specialization.cpp
  Transport.cpp
)

target_include_directories(LoomHardwareRTL PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomHardwareRTL PUBLIC
  CIRCTComb
  CIRCTExportVerilog
  CIRCTHW
  CIRCTSeq
  CIRCTSeqToSV
  CIRCTSV
  CIRCTSupport
  LoomFabricIdentity
  LoomHardwareConfiguration
  LoomHardwareImplementation
  MLIRDataflow
  MLIRIR
  MLIRPass
  LLVMSupport
)
