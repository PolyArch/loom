add_library(LoomRuntimePlatformBinding
  FabricModelPlatform.cpp
  Gem5BridgeABI.cpp
  Gem5BuiltinModels.cpp
  Gem5SimulationBinding.cpp
  Gem5SimulationBindingCodec.cpp
  Gem5SpatialChannel.cpp
  OrderedChannelABI.cpp
  RuntimePlatformBinding.cpp
  RuntimeProvider.cpp
)

target_include_directories(LoomRuntimePlatformBinding PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomRuntimePlatformBinding PUBLIC
  LoomCommon
  LoomFabricArtifact
  LoomHardwareConfiguration
  LoomHardwareImplementation
  LLVMSupport
)

add_library(LoomRuntimeDeploymentLoader
  DeploymentLoader.cpp
  Providers/InProcessPlatform.cpp
)

target_include_directories(LoomRuntimeDeploymentLoader PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomRuntimeDeploymentLoader PUBLIC
  LoomCommon
  LoomDeployment
  LoomFabricArtifact
  LoomFrontendExecutable
  LoomHardwareConfiguration
  LoomHardwareImplementation
  LoomHardwareRTL
  LoomMappingArtifact
  LoomRuntimePlatformBinding
  LLVMSupport
)

add_library(LoomGem5SystemExecution
  Gem5SystemExecution.cpp
  Gem5SystemExecutionFacts.cpp
)

target_include_directories(LoomGem5SystemExecution PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_compile_definitions(LoomGem5SystemExecution PRIVATE
  LOOM_GEM5_CONFIG_SCRIPT_PATH="${LOOM_SOURCE_DIR}/runtime/gem5/configure_loom_system.py"
  LOOM_GEM5_DFG_ENGINE_PATH="$<TARGET_FILE:loom-gem5-dfg-engine>"
  LOOM_GEM5_CGRA_ENGINE_PATH="$<TARGET_FILE:loom-gem5-cgra-engine>"
  LOOM_GEM5_RTL_ENGINE_SOURCE_PATH="${LOOM_SOURCE_DIR}/runtime/gem5/loom_gem5_rtl_engine.cpp"
  LOOM_GEM5_BRIDGE_HEADER_PATH="${LOOM_SOURCE_DIR}/include/Runtime/Gem5BridgeWire.h"
  LOOM_GEM5_CHANNEL_PLAN_HEADER_PATH="${LOOM_SOURCE_DIR}/include/Runtime/Gem5SpatialChannelPlan.h"
  LOOM_GEM5_INVOCATION_WIRE_HEADER_PATH="${LOOM_SOURCE_DIR}/include/Runtime/SpatialInvocationWire.h"
)

add_dependencies(LoomGem5SystemExecution
  loom-gem5-dfg-engine
  loom-gem5-cgra-engine
)

target_link_libraries(LoomGem5SystemExecution PUBLIC
  LoomCommon
  LoomDeployment
  LoomEvaluation
  LoomExternalTool
  LoomMappedRtlExecution
  LoomMappingArtifact
  LoomRuntimePlatformBinding
  LoomSimulator
  MLIRDataflow
  LLVMSupport
)
