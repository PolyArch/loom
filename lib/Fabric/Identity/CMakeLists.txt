add_mlir_library(LoomFabricIdentity
  FabricRefs.cpp
  FabricRefText.cpp
  FabricRefBytes.cpp
  FabricRefImport.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric/Identity

  LINK_LIBS PUBLIC
  LoomCommon
  LoomFabricResourceContract
  LLVMSupport
)
