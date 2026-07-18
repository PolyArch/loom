add_mlir_library(LoomCommon
  ArtifactFinalizer.cpp
  ArtifactStore.cpp
  ArtifactText.cpp
  HwShareGroup.cpp
  IndexWidth.cpp
  LoomConstants.cpp
  ResolvedConfig.cpp
  SynthConfig.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Common

  LINK_LIBS PUBLIC
  LLVMSupport
  MLIRSupport
)
