add_library(LoomEDASynopsys
  Common.cpp
  DesignCompiler.cpp
  FusionCompiler.cpp
  PrimePower.cpp
  PrimeTime.cpp
  Vcs.cpp
)

target_include_directories(LoomEDASynopsys PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomEDASynopsys PUBLIC
  LoomEvaluation
  LoomExternalTool
  LoomHardwareImplementation
  LoomImplementationPlatform
  LLVMSupport
)
