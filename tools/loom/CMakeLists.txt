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

add_custom_target(check-loom-adg
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/adg_test.sh"
    "${LOOM_RUNTIME_DIR}/loom"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  DEPENDS loom loom-sdk
)

add_custom_target(check-loom-llvm-roundtrip
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/ll_roundtrip.sh"
    "${LOOM_RUNTIME_DIR}/loom" --run
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  DEPENDS loom
)

add_custom_target(check-loom-mlir-roundtrip
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/mlir_roundtrip.sh"
    "${LOOM_RUNTIME_DIR}/loom" --run
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  DEPENDS loom mlir-opt mlir-translate
)

add_custom_target(check-loom-scf-roundtrip
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/scf_roundtrip.sh"
    "${LOOM_RUNTIME_DIR}/loom" --run
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  DEPENDS loom mlir-opt mlir-translate
)

add_custom_target(check-loom-handshake
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/handshake.sh"
    "${LOOM_RUNTIME_DIR}/loom"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  DEPENDS loom
)

add_custom_target(check-loom-handshake-ops-stat
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/handshake_ops_stat.sh"
    "${LOOM_RUNTIME_DIR}/loom"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  DEPENDS loom
)

add_custom_target(check-loom-app
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/ll_roundtrip.sh" "${LOOM_RUNTIME_DIR}/loom" --run
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/mlir_roundtrip.sh" "${LOOM_RUNTIME_DIR}/loom" --run
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/scf_roundtrip.sh" "${LOOM_RUNTIME_DIR}/loom" --run
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/handshake_ops_stat.sh" "${LOOM_RUNTIME_DIR}/loom"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  DEPENDS loom mlir-opt mlir-translate
)

add_custom_target(check-loom-fabric-tdd
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/fabric_tdd.sh"
    "${LOOM_RUNTIME_DIR}/loom"
    "${LLVM_SOURCE_DIR}/utils/lit/lit.py"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  DEPENDS loom FileCheck not
)

add_custom_target(check-loom-fabric
  DEPENDS check-loom-fabric-tdd
)

add_custom_target(check-loom
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/check_loom.sh"
    "${LOOM_RUNTIME_DIR}/loom"
    "${LLVM_SOURCE_DIR}/utils/lit/lit.py"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  DEPENDS loom loom-sdk mlir-opt mlir-translate FileCheck not
)

add_custom_target(clean-loom
  COMMAND "${CMAKE_COMMAND}"
    -DLOOM_BINARY="${LOOM_RUNTIME_DIR}/loom"
    -DBUILD_DIR="${CMAKE_BINARY_DIR}"
    -P "${CMAKE_SOURCE_DIR}/tools/loom/clean-loom.cmake"
  COMMAND "${CMAKE_COMMAND}"
    -DSOURCE_DIR="${CMAKE_SOURCE_DIR}"
    -P "${CMAKE_SOURCE_DIR}/tests/cmake/clean-test-output.cmake"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
