set(LOOM_SOURCES
  loom.cpp
  loom_args.cpp
)

add_executable(loom ${LOOM_SOURCES})

set_target_properties(loom PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${LOOM_RUNTIME_DIR}")

llvm_map_components_to_libnames(LOOM_LLVM_LIBS
  core
  support
  linker
  analysis
  native
  nativecodegen
  x86info
  x86codegen
  x86asmparser
)

set(LOOM_MLIR_LIBS
  MLIRControlFlowToSCF
  MLIRParser
  MLIRTargetLLVMIRImport
  MLIRLLVMDialect
  MLIRDLTIDialect
  MLIRUBDialect
  MLIRFuncInlinerExtension
  MLIRIR
  MLIRSupport
)

foreach(target_name IN ITEMS loom)
  target_include_directories(${target_name} PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${LLVM_SOURCE_DIR}/include"
    "${LLVM_BINARY_DIR}/include"
    "${CLANG_SOURCE_DIR}/include"
    "${CLANG_BINARY_DIR}/include"
    "${MLIR_SOURCE_DIR}/include"
    "${MLIR_BINARY_DIR}/include"
    "${CIRCT_SOURCE_DIR}/include"
    "${CIRCT_BINARY_DIR}/include"
  )

  target_link_libraries(${target_name} PRIVATE
    clangFrontend
    clangFrontendTool
    clangCodeGen
    clangTooling
    clangDriver
    clangSerialization
    clangParse
    clangSema
    clangAST
    clangLex
    clangBasic
    LoomConversion
    LoomFabric
    LoomADG
    LoomMapperCore
    LoomViz
    ${LOOM_MLIR_LIBS}
    ${LOOM_LLVM_LIBS}
  )

  target_compile_definitions(${target_name} PRIVATE
    LOOM_ADG_INCLUDE_DIR="${CMAKE_SOURCE_DIR}/include"
    LOOM_ADG_LIB_DIR="${CMAKE_BINARY_DIR}/lib"
  )

  llvm_update_compile_flags(${target_name})
endforeach()

add_dependencies(loom loom_clang_resource)
add_dependencies(loom clang-headers)
add_dependencies(loom loom-sdk)
