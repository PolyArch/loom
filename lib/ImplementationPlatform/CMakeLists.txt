add_mlir_library(LoomImplementationPlatform
  TechnologyCorner.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/ImplementationPlatform

  LINK_LIBS PUBLIC
  LoomCommon
  LLVMSupport
)
