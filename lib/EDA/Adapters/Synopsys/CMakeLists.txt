add_library(LoomEDASynopsys
  Common.cpp
  DesignCompiler.cpp
  DesignCompilerBlock.cpp
  DesignCompilerGenerator.cpp
  DesignCompilerImplementation.cpp
  FusionCompiler.cpp
  FusionCompilerGenerator.cpp
  PrimePower.cpp
  PrimeTime.cpp
  Vcs.cpp
)

target_include_directories(LoomEDASynopsys PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomEDASynopsys PUBLIC
  LoomDSECandidateGenerator
  LoomEDAAsicContracts
  LoomEvaluation
  LoomExternalTool
  LoomHardwareImplementation
  LoomHardwareRTL
  LoomImplementationPlatform
  LLVMSupport
)
