add_library(LoomHardwareImplementation
  ExternalImplementationContract.cpp
  HardwareImplementation.cpp
  HardwareImplementationValidation.cpp
  ImplementationPayload.cpp
  RepresentationFormat.cpp
  RepresentationLocator.cpp
)

target_include_directories(LoomHardwareImplementation PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomHardwareImplementation PUBLIC
  LoomCommon
  LoomFabricArtifact
  LoomHardwareConfiguration
  LoomImplementationPlatform
  LLVMSupport
)
