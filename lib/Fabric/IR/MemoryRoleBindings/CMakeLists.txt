add_mlir_library(LoomFabricMemoryRoleBindings
  MemoryRoleBindings.cpp

  DEPENDS
  MLIRFabricEnumsIncGen

)

target_link_libraries(LoomFabricMemoryRoleBindings PUBLIC
  LLVM
  LoomFabricIdentity
  MLIRDataflow
)
