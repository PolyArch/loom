add_mlir_dialect_library(FccConversion
  HostCodeGen.cpp
  LLVMToCF.cpp
  LLVMToCFTypes.cpp
  MarkDFGDomain.cpp
  SCFPostProcess.cpp
  SCFToDFG.cpp

  LINK_LIBS PUBLIC
    FccDataflow
    FccFabric
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

set_target_properties(FccConversion PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(obj.FccConversion PROPERTIES POSITION_INDEPENDENT_CODE ON)
