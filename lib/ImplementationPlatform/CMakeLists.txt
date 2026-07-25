add_mlir_library(LoomImplementationPlatform
  TechnologyCorner.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/ImplementationPlatform

  LINK_LIBS PUBLIC
  LoomCommon
  LLVMSupport
)

target_include_directories(LoomImplementationPlatform PRIVATE
  ${LOOM_SOURCE_DIR}/lib
)
