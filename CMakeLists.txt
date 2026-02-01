cmake_minimum_required(VERSION 3.20)

project(loom LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(LLVM_ENABLE_PROJECTS "clang" CACHE STRING "")
set(LLVM_TARGETS_TO_BUILD "host" CACHE STRING "")

add_subdirectory(externals/llvm-project/llvm)

list(APPEND CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/externals/llvm-project/llvm/cmake/modules")
include(AddLLVM)

set(LLVM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/externals/llvm-project/llvm")
set(LLVM_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/externals/llvm-project/llvm")
set(CLANG_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/externals/llvm-project/clang")
set(CLANG_BINARY_DIR "${LLVM_BINARY_DIR}/tools/clang")

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
