add_mlir_library(LoomFrontendLowering
  GraphParallelLowering.cpp
  GraphRegionLowering.cpp
  GraphStreamBoundaryLowering.cpp
  GraphIndexLowering.cpp
  RankedMemRefLowering.cpp
  ExpandGraphMemrefCopyPass.cpp
  LowerForallToThreadPass.cpp
  LowerForToGraphPass.cpp
  LowerGraphConstantsPass.cpp
  LowerGraphMemoryPass.cpp
  Pipeline.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Frontend/Lowering

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDataflow
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRLoom
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  MLIRTransforms
  MLIRUBDialect
)
