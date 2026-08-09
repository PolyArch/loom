find_package(Threads REQUIRED)

add_mlir_library(LoomCommon
  ArtifactFinalizer.cpp
  ArtifactLocalReference.cpp
  ArtifactStore.cpp
  ArtifactText.cpp
  BlobDigest.cpp
  BlobStore.cpp
  CanonicalRelation.cpp
  ComponentViewDigest.cpp
  ExternalFileFingerprint.cpp
  IndexWidth.cpp
  MappingDebugLog.cpp
  PointerLayout.cpp
  ResolvedPnrPolicy.cpp
  SpecialMathAccuracy.cpp
  VectorWidth.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Common

)

target_link_libraries(LoomCommon PUBLIC Threads::Threads)
mlir_target_link_libraries(LoomCommon PUBLIC
  LLVMSupport
  LLVMCore
  MLIRDataLayoutInterfaces
  MLIRIR
  MLIRSupport
)
