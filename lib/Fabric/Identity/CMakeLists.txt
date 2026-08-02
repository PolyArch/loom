add_mlir_library(LoomFabricIdentity
  FabricRefs.cpp
  FabricRefText.cpp
  FabricRefBytes.cpp
  FabricRefImport.cpp
  FabricTraversalProjection.cpp
  FabricRefValidation.cpp
  FabricOpCapabilityView.cpp
  FabricFuCapabilityTemplate.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric/Identity

)

target_link_libraries(LoomFabricIdentity PUBLIC
  LLVM
  LoomCommon
  LoomFabricImplementationFamily
  LoomFabricResourceContract
)
