add_executable(loom
  loom.cpp
  loom_args.cpp
  mapper_config.cpp
  loom_runtime.cpp
  loom_pipeline.cpp
)

target_include_directories(loom PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CLANG_INCLUDE_DIRS}
)

# LLVM libraries for IR reading and target support
llvm_map_components_to_libnames(LLVM_LIBS
  core support irreader linker
  native nativecodegen
  ${LLVM_TARGETS_TO_BUILD}
)

target_link_libraries(loom PRIVATE
  # LOOM libraries
  LoomADG
  LoomMapper
  LoomSimulator
  LoomConversion
  LoomDataflow
  LoomFabric
  LoomSVGen
  LoomViz

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

  # MLIR dialects
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
  ${LLVM_LIBS}
)

# Embed the clang resource dir path at compile time
target_compile_definitions(loom PRIVATE
  LOOM_CLANG_RESOURCE_DIR="${LOOM_RESOURCE_DIR}"
  LOOM_SOURCE_DIR="${LOOM_SOURCE_DIR}"
)

if(TARGET clang-resource-headers)
  add_dependencies(loom clang-resource-headers)
endif()
if(TARGET clang-headers)
  add_dependencies(loom clang-headers)
endif()

set_target_properties(loom PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)
