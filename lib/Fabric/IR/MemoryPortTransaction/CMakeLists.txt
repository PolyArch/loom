add_mlir_library(LoomFabricMemoryPortTransaction
  MemoryPortTransaction.cpp

)

target_link_libraries(LoomFabricMemoryPortTransaction PUBLIC
  LLVM
  LoomFabricIdentity
  LoomFabricResourceContract
  MLIRDataflow
)
