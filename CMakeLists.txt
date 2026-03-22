cmake_minimum_required(VERSION 3.24)
project(loom LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(LOOM_ENABLE_ORTOOLS "Enable OR-Tools backed CP-SAT mapper stages" ON)

# Use ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

# Use clang if available
find_program(CLANG_C_COMPILER clang)
find_program(CLANG_CXX_COMPILER clang++)
if(CLANG_C_COMPILER AND CLANG_CXX_COMPILER AND NOT CMAKE_C_COMPILER)
  set(CMAKE_C_COMPILER ${CLANG_C_COMPILER})
  set(CMAKE_CXX_COMPILER ${CLANG_CXX_COMPILER})
endif()

# LLVM/MLIR configuration
set(LLVM_ENABLE_PROJECTS "clang;mlir" CACHE STRING "" FORCE)
set(LLVM_TARGETS_TO_BUILD "X86;RISCV" CACHE STRING "" FORCE)
set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "" FORCE)
set(LLVM_ENABLE_RTTI ON CACHE BOOL "" FORCE)
set(LLVM_ENABLE_EH ON CACHE BOOL "" FORCE)
set(LLVM_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLVM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(MLIR_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "" FORCE)

# CIRCT as external LLVM project
set(LLVM_EXTERNAL_PROJECTS "circt" CACHE STRING "" FORCE)
set(LLVM_EXTERNAL_CIRCT_SOURCE_DIR "${CMAKE_SOURCE_DIR}/externals/circt" CACHE STRING "" FORCE)
set(CIRCT_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(CIRCT_BUILD_TOOLS ON CACHE BOOL "" FORCE)
set(CIRCT_ENABLE_ESI_RUNTIME OFF CACHE BOOL "" FORCE)

# Key directory variables
set(LLVM_SOURCE_DIR "${CMAKE_SOURCE_DIR}/externals/llvm/llvm")
set(LLVM_BINARY_DIR "${CMAKE_BINARY_DIR}/externals/llvm/llvm")
set(MLIR_SOURCE_DIR "${CMAKE_SOURCE_DIR}/externals/llvm/mlir")
set(MLIR_BINARY_DIR "${LLVM_BINARY_DIR}/tools/mlir")
set(CIRCT_SOURCE_DIR "${CMAKE_SOURCE_DIR}/externals/circt")
set(CIRCT_BINARY_DIR "${LLVM_BINARY_DIR}/tools/circt")

# Build externals (LLVM + CIRCT)
add_subdirectory(externals)

# LOOM include paths
set(LOOM_SOURCE_DIR ${CMAKE_SOURCE_DIR})
set(LOOM_BINARY_DIR ${CMAKE_BINARY_DIR})
set(LOOM_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")

# Runtime output directory
set(LOOM_RUNTIME_DIR "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR})

# LLVM/MLIR/CIRCT includes (post-build)
include_directories(${LLVM_SOURCE_DIR}/include)
include_directories(${LLVM_BINARY_DIR}/include)
include_directories(${MLIR_SOURCE_DIR}/include)
include_directories(${MLIR_BINARY_DIR}/include)
include_directories(${CIRCT_SOURCE_DIR}/include)
include_directories(${CIRCT_BINARY_DIR}/include)
include_directories(${LOOM_INCLUDE_DIR})
include_directories(${CMAKE_BINARY_DIR}/include)

# LLVM/MLIR cmake modules
list(APPEND CMAKE_MODULE_PATH "${LLVM_BINARY_DIR}/lib/cmake/llvm")
list(APPEND CMAKE_MODULE_PATH "${MLIR_BINARY_DIR}/lib/cmake/mlir")

# Reuse Clang's generated resource-header directory directly instead of copying
# it into a second build-local location. The extra copy was racy under
# concurrent Ninja invocations and could fail spuriously.
get_target_property(LOOM_CLANG_RESOURCE_INCLUDE_DIR
  clang-resource-headers INTERFACE_INCLUDE_DIRECTORIES)
if(NOT LOOM_CLANG_RESOURCE_INCLUDE_DIR)
  message(FATAL_ERROR "Unable to query clang-resource-headers include dir")
endif()
list(GET LOOM_CLANG_RESOURCE_INCLUDE_DIR 0 LOOM_CLANG_RESOURCE_INCLUDE_DIR)
string(REGEX REPLACE "^\\$<BUILD_INTERFACE:([^>]+)>$" "\\1"
  LOOM_CLANG_RESOURCE_INCLUDE_DIR "${LOOM_CLANG_RESOURCE_INCLUDE_DIR}")
get_filename_component(LOOM_RESOURCE_DIR
  "${LOOM_CLANG_RESOURCE_INCLUDE_DIR}" DIRECTORY)

# Tablegen include directories
set(MLIR_TABLEGEN_FLAGS "-I${LOOM_INCLUDE_DIR}" ${MLIR_TABLEGEN_FLAGS})

# Tablegen targets (generates .inc files under build/include/)
add_subdirectory(include)

# Build loom libraries and tools
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(tests/e2e)
add_subdirectory(tests/unit)
add_subdirectory(tests/rtl)

if(TARGET check-loom)
  add_dependencies(check-loom check-loom-e2e check-loom-gem5 check-loom-rtl)
else()
  add_custom_target(check-loom
    DEPENDS check-loom-unit check-loom-e2e check-loom-gem5 check-loom-rtl
  )
endif()
