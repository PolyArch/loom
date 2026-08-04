add_library(LoomHardwareRTL
  CirctConformance.cpp
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
  MLIRIR
  MLIRPass
  LLVMSupport
)
