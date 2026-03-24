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
  KernelCompiler.cpp
  KernelProfiler.cpp
  L1CoreAssignment.cpp
  L2CoreCompiler.cpp
  MetricsExtractor.cpp
  NoCScheduler.cpp
  PrecompiledKernelLoader.cpp
  ResourceTracker.cpp
  TapestryPipeline.cpp
  TDGLowering.cpp
  TypeAdapters.cpp
)

target_include_directories(LoomSystemCompiler PUBLIC
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
  ${CLANG_INCLUDE_DIRS}
)

# LLVM libraries for IR reading, target support, and optimization passes
# (needed by KernelCompiler for in-process Clang compilation)
llvm_map_components_to_libnames(KERNEL_COMPILER_LLVM_LIBS
  core support irreader
  passes instcombine scalaropts transformutils analysis
  native nativecodegen
  ${LLVM_TARGETS_TO_BUILD}
)

target_link_libraries(LoomSystemCompiler PUBLIC
  LoomADG
  LoomMapper
  LoomConversion
  LoomDataflow
  LoomFabric

  # Clang libraries (for KernelCompiler in-process frontend)
  clangFrontend
  clangFrontendTool
  clangCodeGen
  clangDriver
  clangSerialization
  clangParse
  clangSema
  clangAST
  clangLex
  clangBasic

  CIRCTHandshake

  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRControlFlowToSCF
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRUBDialect
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTargetLLVMIRImport
  MLIRTransforms

  ${KERNEL_COMPILER_LLVM_LIBS}
)

# Embed the clang resource dir path at compile time (for KernelCompiler)
target_compile_definitions(LoomSystemCompiler PRIVATE
  LOOM_CLANG_RESOURCE_DIR="${LOOM_RESOURCE_DIR}"
)

if(TARGET clang-resource-headers)
  add_dependencies(LoomSystemCompiler clang-resource-headers)
endif()
if(TARGET clang-headers)
  add_dependencies(LoomSystemCompiler clang-headers)
endif()

if(LOOM_ENABLE_ORTOOLS AND TARGET ortools::ortools)
  target_link_libraries(LoomSystemCompiler PUBLIC ortools::ortools)
  target_compile_definitions(LoomSystemCompiler PRIVATE LOOM_HAVE_ORTOOLS=1)
endif()
