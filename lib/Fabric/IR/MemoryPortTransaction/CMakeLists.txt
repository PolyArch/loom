add_mlir_library(LoomFabricMemoryPortTransaction
  MemoryPortTransaction.cpp

  LINK_LIBS PUBLIC
  LoomFabricIdentity
  LoomFabricResourceContract
  MLIRDataflow
  LLVMSupport
)
