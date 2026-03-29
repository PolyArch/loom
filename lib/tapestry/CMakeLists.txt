add_library(TapestryCore STATIC
  task_graph.cpp
  derived_metrics.cpp
  tdg_emitter.cpp
  compile.cpp
)

target_include_directories(TapestryCore PUBLIC
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

llvm_map_components_to_libnames(TAPESTRY_CORE_LLVM_LIBS
  core support
)

target_link_libraries(TapestryCore PUBLIC
  LoomTDG
  LoomSystemCompiler

  MLIRIR
  MLIRParser
  MLIRSupport

  ${TAPESTRY_CORE_LLVM_LIBS}
)

set_target_properties(TapestryCore PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(TapestryOptimizer STATIC
  tdg_optimizer.cpp
  host_scheduler.cpp
  co_optimizer.cpp
  ParetoTracker.cpp
  ablation_runner.cpp
)

target_include_directories(TapestryOptimizer PUBLIC
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(TapestryOptimizer PUBLIC
  LoomSystemCompiler

  MLIRIR
  MLIRSupport
)

set_target_properties(TapestryOptimizer PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(TapestryAutoAnalyze STATIC
  auto_analyze.cpp
)

target_include_directories(TapestryAutoAnalyze PUBLIC
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
  ${CLANG_INCLUDE_DIRS}
)

llvm_map_components_to_libnames(TAPESTRY_AA_LLVM_LIBS
  core support irreader linker
  passes instcombine scalaropts transformutils analysis
  native nativecodegen
  ${LLVM_TARGETS_TO_BUILD}
)

target_link_libraries(TapestryAutoAnalyze PUBLIC
  LoomConversion
  LoomDataflow
  LoomFabric

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

  MLIRDLTIDialect
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRControlFlowToSCF
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

  ${TAPESTRY_AA_LLVM_LIBS}
)

target_compile_definitions(TapestryAutoAnalyze PRIVATE
  LOOM_CLANG_RESOURCE_DIR="${LOOM_RESOURCE_DIR}"
)

if(TARGET clang-resource-headers)
  add_dependencies(TapestryAutoAnalyze clang-resource-headers)
endif()
if(TARGET clang-headers)
  add_dependencies(TapestryAutoAnalyze clang-headers)
endif()
