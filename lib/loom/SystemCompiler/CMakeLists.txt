add_library(LoomSystemCompiler STATIC
  ArchitectureFactory.cpp
  BendersDriver.cpp
  PrecompiledKernelLoader.cpp
  TDGLowering.cpp
)

target_include_directories(LoomSystemCompiler PUBLIC
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(LoomSystemCompiler PUBLIC
  LoomADG
  LoomMapper
  LoomConversion
  LoomDataflow
  LoomFabric

  CIRCTHandshake

  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRControlFlowToSCF
  MLIRFuncDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
)
