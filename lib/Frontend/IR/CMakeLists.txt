add_mlir_dialect_library(MLIRLoom
  LoomDialect.cpp
  LoomOps.cpp
  StructuredProgramArtifact.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Frontend

  DEPENDS
  MLIRLoomDialectIncGen
  MLIRLoomOpsIncGen

  LINK_LIBS PUBLIC
  LoomCommon
  MLIRDataflow
  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRParser
  MLIRRegisterAllDialects
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRSupport
)
