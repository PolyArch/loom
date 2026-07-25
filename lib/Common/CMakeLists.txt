find_package(Threads REQUIRED)

add_mlir_library(LoomCommon
  ArtifactFinalizer.cpp
  ArtifactLocalReference.cpp
  ArtifactStore.cpp
  ArtifactText.cpp
  BlobDigest.cpp
  BlobStore.cpp
  ComponentViewDigest.cpp
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
  Threads::Threads
)
