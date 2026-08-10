add_library(LoomHardwareConfiguration
  ConfigurationABI.cpp
  PackedConfigurationABI.cpp
  ProgrammingUnitRef.cpp
)

target_include_directories(LoomHardwareConfiguration PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomHardwareConfiguration PUBLIC
  LoomCommon
  LoomFabricArtifact
  LoomFabricIdentity
  LLVMSupport
)
