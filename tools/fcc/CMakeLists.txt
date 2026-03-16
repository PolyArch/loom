add_executable(fcc
  fcc.cpp
  fcc_args.cpp
  fcc_pipeline.cpp
)

target_include_directories(fcc PRIVATE
  ${FCC_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  # Clang headers
  ${CMAKE_SOURCE_DIR}/externals/llvm/clang/include
  ${LLVM_BINARY_DIR}/tools/clang/include
)

# LLVM libraries for IR reading and target support
llvm_map_components_to_libnames(LLVM_LIBS
  core support irreader linker
  native nativecodegen
  ${LLVM_TARGETS_TO_BUILD}
)

target_link_libraries(fcc PRIVATE
  # FCC libraries
  FccADG
  FccMapper
  FccConversion
  FccDataflow
  FccFabric
  FccViz

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
target_compile_definitions(fcc PRIVATE
  FCC_CLANG_RESOURCE_DIR="${FCC_RESOURCE_DIR}"
)

add_dependencies(fcc fcc_clang_resource clang-headers)

set_target_properties(fcc PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${FCC_RUNTIME_DIR}
)
