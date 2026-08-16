add_library(LoomApplicationPortfolio
  Manifest.cpp
  SourceAdmission.cpp
)

target_include_directories(LoomApplicationPortfolio PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomApplicationPortfolio PUBLIC
  LoomCommon
  LLVMSupport
)

add_library(LoomApplicationBuild
  Build.cpp
  BuildDiagnostics.cpp
  ExecutionGlue.cpp
)

target_include_directories(LoomApplicationBuild PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomApplicationBuild PUBLIC
  LoomCommon
  LoomDeployment
  LoomDSE
  LoomDSECompilerGenerator
  LoomDSEProductionOwners
  LoomFabricArtifact
  LoomFrontendCompilation
  LoomFrontendExecutable
  LoomFrontendExecutableLinker
  LoomHardwareConfiguration
  LoomHardwareImplementation
  LoomMappingArtifact
  LoomRuntimePlatformBinding
  LoomSimulator
  LLVMCore
  LLVMTransformUtils
  LLVMSupport
)
