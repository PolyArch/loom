add_mlir_library(LoomCommon
  Config.cpp
  HwShareGroup.cpp
  IndexWidth.cpp
  SynthConfig.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Common

  LINK_LIBS PUBLIC
  MLIRSupport
)
