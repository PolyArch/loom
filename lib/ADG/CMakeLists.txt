add_mlir_library(LoomADGBuilder
  Builder.cpp
  BuilderBuiltins.cpp
  BuilderRecipes.cpp
  BuilderSharedMemoryRecipes.cpp
  BuilderSharedReductionCompute.cpp
  BuilderSharedReductionRouting.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/ADG

  LINK_LIBS PUBLIC
  LLVMSupport
)
