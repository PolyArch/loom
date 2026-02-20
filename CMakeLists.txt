cmake_minimum_required(VERSION 3.20)

# Prefer clang/clang++ if available and no compiler was explicitly specified.
if(NOT DEFINED CMAKE_C_COMPILER AND NOT DEFINED ENV{CC})
  find_program(_LOOM_CLANG clang)
  if(_LOOM_CLANG)
    set(CMAKE_C_COMPILER "${_LOOM_CLANG}")
  endif()
  unset(_LOOM_CLANG CACHE)
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
  find_program(_LOOM_CLANGXX clang++)
  if(_LOOM_CLANGXX)
    set(CMAKE_CXX_COMPILER "${_LOOM_CLANGXX}")
  endif()
  unset(_LOOM_CLANGXX CACHE)
endif()

# Use ccache if available to speed up rebuilds.
if(NOT DEFINED CMAKE_C_COMPILER_LAUNCHER AND NOT DEFINED CMAKE_CXX_COMPILER_LAUNCHER)
  find_program(_LOOM_CCACHE ccache)
  if(_LOOM_CCACHE)
    set(CMAKE_C_COMPILER_LAUNCHER "${_LOOM_CCACHE}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${_LOOM_CCACHE}")
  endif()
  unset(_LOOM_CCACHE CACHE)
endif()

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

set(LOOM_SV_TEMPLATE_DIR "${CMAKE_BINARY_DIR}/lib/sv")

add_custom_target(loom_sv_templates ALL
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${LOOM_SV_TEMPLATE_DIR}"
  COMMAND "${CMAKE_COMMAND}" -E copy_directory
          "${CMAKE_SOURCE_DIR}/lib/loom/Hardware/SystemVerilog"
          "${LOOM_SV_TEMPLATE_DIR}"
)

add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(tests)
