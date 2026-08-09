add_library(LoomEDASynopsys
  Common.cpp
  DesignCompiler.cpp
  DesignCompilerGenerator.cpp
  FusionCompiler.cpp
  PrimePower.cpp
  PrimeTime.cpp
  Vcs.cpp
)

target_include_directories(LoomEDASynopsys PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomEDASynopsys PUBLIC
  LoomDSECandidateGenerator
  LoomEvaluation
  LoomExternalTool
  LoomHardwareImplementation
  LoomImplementationPlatform
  LLVMSupport
)
