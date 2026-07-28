set(LLVM_LINK_COMPONENTS
  Core
  IRReader
  Support
)

add_llvm_executable(loom-dfg-run
  loom-dfg-run.cpp
)

llvm_update_compile_flags(loom-dfg-run)

set_target_properties(loom-dfg-run PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_link_libraries(loom-dfg-run PRIVATE
  LoomADGBuilder
  LoomCommon
  LoomDSE
  LoomFrontendCompilation
  LoomNativeSimulationOracle
  LoomSimulator
)

mlir_check_all_link_libraries(loom-dfg-run)
