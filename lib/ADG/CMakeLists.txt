add_mlir_library(LoomADGBuilder
  Builder.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/ADG

  LINK_LIBS PUBLIC
  LLVMSupport
)
