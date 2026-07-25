add_mlir_library(LoomADGBuilder
  Builder.cpp
  BuilderBuiltins.cpp
  BuilderCapabilities.cpp
  BuilderRecipes.cpp
  BuilderSharedMemoryRecipes.cpp
  BuilderSharedReductionCompute.cpp
  BuilderSharedReductionRouting.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/ADG

  DEPENDS
  MLIRDataflowEnumsIncGen
  MLIRFabricEnumsIncGen

  LINK_LIBS PUBLIC
  LLVMSupport
  MLIRArithDialect
  MLIRDataflowEnums
  MLIRFabric
  MLIRFabricEnums
  MLIRIR
  MLIRParser
)
