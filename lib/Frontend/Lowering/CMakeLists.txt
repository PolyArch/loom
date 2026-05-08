add_mlir_library(LoomFrontendLowering
  LowerForallToThreadPass.cpp
  LowerForToGraphPass.cpp
  LowerReductionToStreamPass.cpp
  Pipeline.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Frontend/Lowering

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDataflow
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  MLIRTransforms
  MLIRUBDialect
)
