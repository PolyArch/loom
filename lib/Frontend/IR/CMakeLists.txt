add_mlir_dialect_library(MLIRLoom
  LoomDialect.cpp
  LoomOps.cpp
  StructuredProgramArtifact.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Frontend

  DEPENDS
  MLIRLoomDialectIncGen
  MLIRLoomOpsIncGen

)

target_link_libraries(MLIRLoom PUBLIC
  LoomCommon
  MLIRDataflow
)
mlir_target_link_libraries(MLIRLoom PUBLIC
  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRParser
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRToLLVMIRTranslationRegistration
)
