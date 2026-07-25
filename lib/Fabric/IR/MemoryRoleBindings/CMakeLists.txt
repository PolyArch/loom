add_mlir_library(LoomFabricMemoryRoleBindings
  MemoryRoleBindings.cpp

  DEPENDS
  MLIRFabricEnumsIncGen

  LINK_LIBS PUBLIC
  LoomFabricIdentity
  MLIRDataflow
  LLVMSupport
)
