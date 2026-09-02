set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_executable(loom-system-run
  loom-system-run.cpp
)

llvm_update_compile_flags(loom-system-run)

set_target_properties(loom-system-run PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_link_libraries(loom-system-run PRIVATE
  LoomApplicationDeploymentRuntime
  LoomApplicationRuntimeManifest
  LoomDSEProductionOwners
  LoomEDAOpenSource
  LoomGem5SystemExecution
  LoomRuntimeResourceTime
  LoomRuntimePlatformBinding
  LoomStructuredFabricAnalyticModel
)

mlir_check_all_link_libraries(loom-system-run)
