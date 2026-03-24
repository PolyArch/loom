add_library(LoomSystemCompiler STATIC
  ArchitectureFactory.cpp
  BendersDriver.cpp
  BendersHelpers.cpp
  BufferAllocator.cpp
  Contract.cpp
  CostSummary.cpp
  DMAScheduler.cpp
  ExecutionModel.cpp
  FailureAnalyzer.cpp
  FlatCompilerBaselines.cpp
  InfeasibilityCut.cpp
  KernelProfiler.cpp
  L1CoreAssignment.cpp
  L2CoreCompiler.cpp
  MetricsExtractor.cpp
  NoCScheduler.cpp
  PrecompiledKernelLoader.cpp
  ResourceTracker.cpp
  TapestryPipeline.cpp
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

if(LOOM_ENABLE_ORTOOLS AND TARGET ortools::ortools)
  target_link_libraries(LoomSystemCompiler PUBLIC ortools::ortools)
  target_compile_definitions(LoomSystemCompiler PRIVATE LOOM_HAVE_ORTOOLS=1)
endif()
