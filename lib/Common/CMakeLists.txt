add_mlir_library(LoomCommon
  ArtifactFinalizer.cpp
  ArtifactStore.cpp
  ArtifactText.cpp
  ComponentViewDigest.cpp
  HwShareGroup.cpp
  IndexWidth.cpp
  LoomConstants.cpp
  ResolvedConfig.cpp
  SynthConfig.cpp
  VectorWidth.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Common

  LINK_LIBS PUBLIC
  LLVMSupport
  MLIRDataLayoutInterfaces
  MLIRIR
  MLIRSupport
)
