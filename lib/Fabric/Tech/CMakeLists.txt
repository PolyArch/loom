add_mlir_library(MLIRFabricTech
  CostModel.cpp
  Parallel.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric/Tech

  LINK_LIBS PUBLIC
  LoomCommon
  MLIRArithDialect
  MLIRFabric
  MLIRIR
  MLIRSupport
  LLVMSupport
)
