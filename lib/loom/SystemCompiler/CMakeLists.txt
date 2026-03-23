add_library(LoomSystemCompiler
  Contract.cpp
  InfeasibilityCut.cpp
  CostSummary.cpp
  L2CoreCompiler.cpp
  FailureAnalyzer.cpp
  MetricsExtractor.cpp
  ResourceTracker.cpp
  L1CoreAssignment.cpp
  KernelProfiler.cpp
  FlatCompilerBaselines.cpp
  NoCScheduler.cpp
  BufferAllocator.cpp
  DMAScheduler.cpp
  BendersDriver.cpp
  BendersHelpers.cpp
  TDGLowering.cpp
  TapestryPipeline.cpp
)

target_include_directories(LoomSystemCompiler PUBLIC
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

llvm_map_components_to_libnames(LOOM_SC_LLVM_LIBS support)

target_link_libraries(LoomSystemCompiler PUBLIC
  ${LOOM_SC_LLVM_LIBS}
  LoomMapper
  LoomMultiCoreSim
  LoomSVGen
  MLIRIR
  MLIRParser
  MLIRSupport
)

if(LOOM_ENABLE_ORTOOLS AND TARGET ortools::ortools)
  target_link_libraries(LoomSystemCompiler PUBLIC ortools::ortools)
  target_compile_definitions(LoomSystemCompiler PRIVATE LOOM_HAVE_ORTOOLS=1)
  target_compile_features(LoomSystemCompiler PRIVATE cxx_std_20)
endif()

set_target_properties(LoomSystemCompiler PROPERTIES POSITION_INDEPENDENT_CODE ON)
