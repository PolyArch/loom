add_mlir_library(LoomFabricSystemContracts
  FabricSystemContracts.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric/Artifact

)

target_link_libraries(LoomFabricSystemContracts PUBLIC
  LLVM
  LoomCommon
  LoomFabricIdentity
  LoomFabricResourceContract
)
