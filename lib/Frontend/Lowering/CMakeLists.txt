add_mlir_library(LoomFrontendLowering
  GraphRegionLowering.cpp
  GraphIndexLowering.cpp
  LowerForallToThreadPass.cpp
  LowerForToGraphPass.cpp
  LowerGraphConstantsPass.cpp
  LowerKnownLibraryCallsPass.cpp
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
