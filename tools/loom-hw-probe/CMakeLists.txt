add_llvm_executable(loom-hw-probe
  loom-hw-probe.cpp
)

llvm_update_compile_flags(loom-hw-probe)

set_target_properties(loom-hw-probe PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_link_libraries(loom-hw-probe PRIVATE
  LoomEDAOpenSource
  LoomExternalTool
  LoomHardwareRTL
  LLVMOption
  LLVMSupport
)
