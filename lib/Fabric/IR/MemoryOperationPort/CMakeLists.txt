add_mlir_library(LoomFabricMemoryOperationPort
  MemoryOperationPort.cpp
  MemoryOperationPortCodec.cpp

  LINK_LIBS PUBLIC
  LoomFabricIdentity
  LoomFabricMemoryCapabilityDomains
  LoomFabricMemoryPortTransaction
  LoomFabricResourceContract
  MLIRDataflow
  MLIRFabricEnums
  MLIRIR
  MLIRSupport
  LLVMSupport
)
