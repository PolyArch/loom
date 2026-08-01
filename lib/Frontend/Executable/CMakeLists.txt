add_mlir_library(LoomFrontendExecutable
  CompilerTargetBinding.cpp
  CompilerTargetBindingCodec.cpp
  CompilerTargetCohort.cpp
  CompilerTargetMachine.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Frontend/Executable

  LINK_LIBS PUBLIC
  LoomCommon
  LoomFabricArtifact
  LoomFabricIdentity
  LoomFrontendPayload
  LLVM
)
