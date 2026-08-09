add_llvm_library(LoomEDACadence
  Common.cpp
  Genus.cpp
  Innovus.cpp
  Joules.cpp
  Tempus.cpp
  Voltus.cpp
  Xcelium.cpp

  LINK_LIBS
  LoomCommon
  LoomDSE
  LoomEDAAsicContracts
  LoomExternalTool
  LoomHardwareImplementation
  LoomImplementationPlatform
  LLVMSupport
)
