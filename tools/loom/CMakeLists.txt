set(LOOM_SOURCES
  loom.cpp
)

add_executable(loom ${LOOM_SOURCES})

set_target_properties(loom PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${LOOM_RUNTIME_DIR}")

llvm_map_components_to_libnames(LOOM_LLVM_LIBS
  core
  support
  linker
  analysis
  native
)

set(LOOM_MLIR_LIBS
  MLIRTargetLLVMIRImport
  MLIRLLVMDialect
  MLIRDLTIDialect
  MLIRIR
  MLIRSupport
)

foreach(target_name IN ITEMS loom)
  target_include_directories(${target_name} PRIVATE
    "${LLVM_SOURCE_DIR}/include"
    "${LLVM_BINARY_DIR}/include"
    "${CLANG_SOURCE_DIR}/include"
    "${CLANG_BINARY_DIR}/include"
    "${MLIR_SOURCE_DIR}/include"
    "${MLIR_BINARY_DIR}/include"
  )

  target_link_libraries(${target_name} PRIVATE
    clangFrontend
    clangCodeGen
    clangTooling
    clangDriver
    clangSerialization
    clangParse
    clangSema
    clangAST
    clangLex
    clangBasic
    ${LOOM_MLIR_LIBS}
    ${LOOM_LLVM_LIBS}
  )

  llvm_update_compile_flags(${target_name})
endforeach()

add_dependencies(loom loom_clang_resource)
add_dependencies(loom clang-headers)

add_custom_target(check-loom-llvm-roundtrip
  COMMAND "${CMAKE_SOURCE_DIR}/tools/loom/loom_apps.sh" "${LOOM_RUNTIME_DIR}/loom" --run
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  DEPENDS loom
)

add_custom_target(check-loom-mlir-roundtrip
  COMMAND "${CMAKE_SOURCE_DIR}/tools/loom/loom_mlir_apps.sh" "${LOOM_RUNTIME_DIR}/loom" --run
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  DEPENDS loom mlir-opt mlir-translate
)

add_custom_target(check-loom
  COMMAND "${CMAKE_SOURCE_DIR}/tools/loom/loom_apps.sh" "${LOOM_RUNTIME_DIR}/loom" --run
  COMMAND "${CMAKE_SOURCE_DIR}/tools/loom/loom_mlir_apps.sh" "${LOOM_RUNTIME_DIR}/loom" --run
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  DEPENDS loom mlir-opt mlir-translate
)
