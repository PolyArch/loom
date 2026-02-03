cmake_minimum_required(VERSION 3.20)

project(loom LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(LLVM_ENABLE_PROJECTS "clang;mlir" CACHE STRING "")
set(LLVM_TARGETS_TO_BUILD "host" CACHE STRING "")
set(LLVM_EXTERNAL_PROJECTS "circt" CACHE STRING "")
set(LLVM_EXTERNAL_CIRCT_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/externals/circt" CACHE PATH "")

set(CIRCT_INCLUDE_TESTS OFF CACHE BOOL "")
set(CIRCT_INCLUDE_TOOLS OFF CACHE BOOL "")
set(CIRCT_BUILD_TOOLS OFF CACHE BOOL "")
set(ESI_RUNTIME OFF CACHE BOOL "")

add_subdirectory(externals/llvm-project/llvm)

list(APPEND CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/externals/llvm-project/llvm/cmake/modules"
  "${CMAKE_CURRENT_SOURCE_DIR}/externals/llvm-project/mlir/cmake/modules")
include(AddLLVM)
include(AddMLIR)

set(LLVM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/externals/llvm-project/llvm")
set(LLVM_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/externals/llvm-project/llvm")
set(CLANG_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/externals/llvm-project/clang")
set(CLANG_BINARY_DIR "${LLVM_BINARY_DIR}/tools/clang")
set(MLIR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/externals/llvm-project/mlir")
set(MLIR_BINARY_DIR "${LLVM_BINARY_DIR}/tools/mlir")
set(CIRCT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/externals/circt")
set(CIRCT_BINARY_DIR "${LLVM_BINARY_DIR}/tools/circt")

set(LOOM_RUNTIME_DIR "${CMAKE_BINARY_DIR}/bin")

set(LOOM_RESOURCE_DIR
  "${CMAKE_BINARY_DIR}/lib/clang")

add_custom_target(loom_clang_resource ALL
  COMMAND "${CMAKE_COMMAND}" -E make_directory
          "${LOOM_RESOURCE_DIR}/include"
  COMMAND "${CMAKE_COMMAND}" -E copy_directory
          "${CLANG_SOURCE_DIR}/lib/Headers"
          "${LOOM_RESOURCE_DIR}/include"
  BYPRODUCTS "${LOOM_RESOURCE_DIR}/include/stdarg.h"
)

add_subdirectory(lib)
add_subdirectory(tools)
