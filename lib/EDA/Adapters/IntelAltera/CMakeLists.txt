add_library(LoomEDAIntelAltera
  Quartus.cpp
)

target_include_directories(LoomEDAIntelAltera PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomEDAIntelAltera PUBLIC
  LoomDSECandidateGenerator
  LoomExternalTool
  LoomHardwareImplementation
  LoomImplementationPlatform
  LLVMSupport
)
