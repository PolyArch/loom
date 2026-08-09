add_mlir_library(LoomFabricMemoryOperationPort
  MemoryOperationPort.cpp
  MemoryOperationPortCodec.cpp

  DEPENDS
  LoomOperationSchemasIncGen
  MLIRDataflowAttrsIncGen
  MLIRDataflowDialectIncGen
  MLIRDataflowInterfacesIncGen
  MLIRDataflowOpsIncGen

)

target_link_libraries(LoomFabricMemoryOperationPort PUBLIC
  LLVM
  LoomFabricIdentity
  LoomFabricMemoryCapabilityDomains
  LoomFabricMemoryPortTransaction
  LoomFabricResourceContract
  MLIRDataflow
  MLIRFabricEnums
)
mlir_target_link_libraries(LoomFabricMemoryOperationPort PUBLIC
  MLIRIR
  MLIRSupport
)
