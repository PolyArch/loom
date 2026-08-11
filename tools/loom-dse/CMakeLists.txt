add_llvm_executable(loom-dse loom-dse.cpp)
llvm_update_compile_flags(loom-dse)

set_target_properties(loom-dse PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

target_link_libraries(loom-dse PRIVATE
  LoomDSE
  LoomEDAOpenSource
  LoomStructuredFabricAnalyticModel
  LLVMSupport
)
