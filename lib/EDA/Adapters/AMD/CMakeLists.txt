add_library(LoomEDAAMD
  Vivado.cpp
)

target_include_directories(LoomEDAAMD PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomEDAAMD PUBLIC
  LoomDSECandidateGenerator
  LoomExternalTool
  LoomHardwareImplementation
  LoomImplementationPlatform
  LLVMSupport
)
