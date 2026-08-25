add_mlir_library(LoomFrontendLowering
  CanonicalDataflowLowering.cpp
  ExactMemRefLayout.cpp
  GraphMemoryAddressing.cpp
  GraphParallelLowering.cpp
  GraphRegionAdmission.cpp
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

)

target_link_libraries(LoomFrontendLowering PUBLIC
  LoomFrontendAnalysis
  MLIRDataflow
  MLIRLoom
)
mlir_target_link_libraries(LoomFrontendLowering PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  MLIRTransforms
  MLIRUBDialect
  MLIRVectorDialect
)
