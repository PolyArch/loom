add_mlir_dialect_library(LoomConversion
  HostCodeGen.cpp
  LLVMToCF.cpp
  LLVMToCFTypes.cpp
  MarkDFGDomain.cpp
  SCFPostProcess.cpp
  SCFToDFG.cpp

  LINK_LIBS PUBLIC
    LoomDataflow
    LoomFabric
    CIRCTHandshake
    MLIRArithDialect
    MLIRControlFlowDialect
    MLIRControlFlowToSCF
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRSCFTransforms
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms
)

set_target_properties(LoomConversion PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(obj.LoomConversion PROPERTIES POSITION_INDEPENDENT_CODE ON)
