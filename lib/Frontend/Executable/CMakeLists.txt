add_mlir_library(LoomFrontendExecutable
  CompilerTargetBinding.cpp
  CompilerTargetBindingCodec.cpp
  CompilerTargetCodegen.cpp
  CompilerTargetCohort.cpp
  CompilerTargetMachine.cpp
  ExecutableElf.cpp
  InstructionCoreBinary.cpp
  InstructionCoreBinaryCodec.cpp
  InstructionCoreElf.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Frontend/Executable

  LINK_LIBS PUBLIC
  LoomCommon
  LoomFabricArtifact
  LoomFabricIdentity
  LoomFrontendPayload
  MLIRDataflow
  LLVM
)

add_library(LoomFrontendExecutableLinker
  CompilerTargetLinker.cpp
)

target_include_directories(LoomFrontendExecutableLinker PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomFrontendExecutableLinker PUBLIC
  LoomFrontendExecutable
  lldELF
  LLVMSupport
)
