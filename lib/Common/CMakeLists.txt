find_package(Threads REQUIRED)
# Build requirement: the host OpenSSL libcrypto with development headers. Blob
# digests and framed artifact identities share its SHA-256 implementation over
# the complete byte sequence, including streamed blobs. The LLVM SHA256
# helper narrows its byte count at the 512 MiB length-encoding boundary and is
# therefore not an admissible digest owner.
find_package(OpenSSL REQUIRED COMPONENTS Crypto)

add_mlir_library(LoomCommon
  ArtifactFinalizer.cpp
  ArtifactLocalReference.cpp
  ArtifactStore.cpp
  ArtifactText.cpp
  BlobDigest.cpp
  BlobStore.cpp
  CanonicalRelation.cpp
  ComponentViewDigest.cpp
  DiagnosticVerbosity.cpp
  ExecutionControl.cpp
  ExternalFileFingerprint.cpp
  IndexWidth.cpp
  InvocationDiagnosticLog.cpp
  MappingDebugLog.cpp
  PointerLayout.cpp
  ResolvedPnrPolicy.cpp
  SpecialMathAccuracy.cpp
  VectorWidth.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Common

)

target_link_libraries(LoomCommon PUBLIC OpenSSL::Crypto Threads::Threads)
mlir_target_link_libraries(LoomCommon PUBLIC
  LLVMSupport
  LLVMCore
  MLIRDataLayoutInterfaces
  MLIRIR
  MLIRSupport
)
