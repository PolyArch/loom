add_llvm_executable(loom-backend-tool-catalog
  loom-backend-tool-catalog.cpp
)

llvm_update_compile_flags(loom-backend-tool-catalog)

set_target_properties(loom-backend-tool-catalog PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_link_libraries(loom-backend-tool-catalog PRIVATE
  LoomExternalTool
  LLVMOption
  LLVMSupport
)
