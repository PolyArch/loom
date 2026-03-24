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
  # Loom libraries (reuse existing conversion pipeline)
  LoomConversion
  LoomDataflow
  LoomFabric

  # Clang libraries (in-process frontend)
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

  # CIRCT
  CIRCTHandshake

  # MLIR dialects (DLTIDialect needed for LLVM IR import)
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

  # MLIR infrastructure
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTargetLLVMIRImport
  MLIRTransforms

  # LLVM
  ${TAPESTRY_AA_LLVM_LIBS}
)

# Embed the clang resource dir path at compile time
target_compile_definitions(TapestryAutoAnalyze PRIVATE
  LOOM_CLANG_RESOURCE_DIR="${LOOM_RESOURCE_DIR}"
)

if(TARGET clang-resource-headers)
  add_dependencies(TapestryAutoAnalyze clang-resource-headers)
endif()
if(TARGET clang-headers)
  add_dependencies(TapestryAutoAnalyze clang-headers)
endif()
