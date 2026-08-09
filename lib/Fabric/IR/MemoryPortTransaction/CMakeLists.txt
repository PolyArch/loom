add_mlir_library(LoomFabricMemoryPortTransaction
  MemoryPortTransaction.cpp

  DEPENDS
  LoomOperationSchemasIncGen
  MLIRDataflowAttrsIncGen
  MLIRDataflowDialectIncGen
  MLIRDataflowInterfacesIncGen
  MLIRDataflowOpsIncGen

)

target_link_libraries(LoomFabricMemoryPortTransaction PUBLIC
  LLVM
  LoomFabricIdentity
  LoomFabricResourceContract
  MLIRDataflow
)
