add_llvm_library(LoomADG
  ADGBuilder.cpp
  ADGBuilderGen.cpp
  ADGBuilderPE.cpp
  ADGBuilderTypes.cpp
  ADGBuilderValidation.cpp
  ADGExportSV.cpp

  LINK_LIBS PUBLIC
    LoomFabric
    LoomDataflow
    CIRCTHandshake
    MLIRArithDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRFuncDialect
    MLIRParser
    MLIRIR
    MLIRSupport
)

target_include_directories(LoomADG
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  PRIVATE
    "${LLVM_SOURCE_DIR}/include"
    "${LLVM_BINARY_DIR}/include"
    "${MLIR_SOURCE_DIR}/include"
    "${MLIR_BINARY_DIR}/include"
    "${CIRCT_SOURCE_DIR}/include"
    "${CIRCT_BINARY_DIR}/include"
)

target_compile_definitions(LoomADG PRIVATE
  LOOM_SV_TEMPLATE_DIR="${CMAKE_SOURCE_DIR}/lib/loom/Hardware/SystemVerilog"
)

llvm_update_compile_flags(LoomADG)
set_target_properties(LoomADG PROPERTIES POSITION_INDEPENDENT_CODE ON)
