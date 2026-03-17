cmake_minimum_required(VERSION 3.20)
project(fcc LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
set(CIRCT_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
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

# FCC include paths
set(FCC_SOURCE_DIR ${CMAKE_SOURCE_DIR})
set(FCC_BINARY_DIR ${CMAKE_BINARY_DIR})
set(FCC_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")

# Runtime output directory
set(FCC_RUNTIME_DIR "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${FCC_RUNTIME_DIR})

# LLVM/MLIR/CIRCT includes (post-build)
include_directories(${LLVM_SOURCE_DIR}/include)
include_directories(${LLVM_BINARY_DIR}/include)
include_directories(${MLIR_SOURCE_DIR}/include)
include_directories(${MLIR_BINARY_DIR}/include)
include_directories(${CIRCT_SOURCE_DIR}/include)
include_directories(${CIRCT_BINARY_DIR}/include)
include_directories(${FCC_INCLUDE_DIR})
include_directories(${CMAKE_BINARY_DIR}/include)

# LLVM/MLIR cmake modules
list(APPEND CMAKE_MODULE_PATH "${LLVM_BINARY_DIR}/lib/cmake/llvm")
list(APPEND CMAKE_MODULE_PATH "${MLIR_BINARY_DIR}/lib/cmake/mlir")

# Copy clang built-in headers to build/lib/clang/include/
set(CLANG_SOURCE_DIR "${CMAKE_SOURCE_DIR}/externals/llvm/clang")
set(FCC_RESOURCE_DIR "${CMAKE_BINARY_DIR}/lib/clang")
add_custom_target(fcc_clang_resource ALL
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${FCC_RESOURCE_DIR}/include"
  COMMAND "${CMAKE_COMMAND}" -E copy_directory
          "${CLANG_SOURCE_DIR}/lib/Headers"
          "${FCC_RESOURCE_DIR}/include"
  BYPRODUCTS "${FCC_RESOURCE_DIR}/include/stdarg.h"
)

# Tablegen include directories
set(MLIR_TABLEGEN_FLAGS "-I${FCC_INCLUDE_DIR}" ${MLIR_TABLEGEN_FLAGS})

# Tablegen targets (generates .inc files under build/include/)
add_subdirectory(include)

# Build fcc libraries and tools
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(tests/e2e/common)
add_subdirectory(tests/e2e/vecadd)
add_subdirectory(tests/e2e/vecmul)
add_subdirectory(tests/e2e/vecscale)
add_subdirectory(tests/e2e/axpy)
add_subdirectory(tests/e2e/binary_search)
add_subdirectory(tests/e2e/byte_swap)
add_subdirectory(tests/e2e/bit_reverse)
add_subdirectory(tests/e2e/cumsum)
add_subdirectory(tests/e2e/clz)
add_subdirectory(tests/e2e/compact)
add_subdirectory(tests/e2e/compact_predicate)
add_subdirectory(tests/e2e/compare_swap)
add_subdirectory(tests/e2e/convolve_1d)
add_subdirectory(tests/e2e/ctz)
add_subdirectory(tests/e2e/crc32)
add_subdirectory(tests/e2e/database_join)
add_subdirectory(tests/e2e/autocorrelation)
add_subdirectory(tests/e2e/dotprod)
add_subdirectory(tests/e2e/delta_decode)
add_subdirectory(tests/e2e/delta_encode)
add_subdirectory(tests/e2e/edit_distance_step)
add_subdirectory(tests/e2e/find_first_set)
add_subdirectory(tests/e2e/gf_mul)
add_subdirectory(tests/e2e/gather)
add_subdirectory(tests/e2e/hash_mix)
add_subdirectory(tests/e2e/kmp_table)
add_subdirectory(tests/e2e/lower_bound)
add_subdirectory(tests/e2e/merge)
add_subdirectory(tests/e2e/modexp)
add_subdirectory(tests/e2e/parity)
add_subdirectory(tests/e2e/partition)
add_subdirectory(tests/e2e/pack_bits)
add_subdirectory(tests/e2e/popcount)
add_subdirectory(tests/e2e/prefix_sum_exclusive)
add_subdirectory(tests/e2e/prefix_sum_inclusive)
add_subdirectory(tests/e2e/rotate_bits)
add_subdirectory(tests/e2e/rle_decode)
add_subdirectory(tests/e2e/rle_encode)
add_subdirectory(tests/e2e/string_hash)
add_subdirectory(tests/e2e/string_compare)
add_subdirectory(tests/e2e/sbox_lookup)
add_subdirectory(tests/e2e/stream_nested)
add_subdirectory(tests/e2e/stream_update)
add_subdirectory(tests/e2e/unpack_bits)
add_subdirectory(tests/e2e/upper_bound)
add_subdirectory(tests/e2e/vecsum)
add_subdirectory(tests/e2e/vecsum_temporal)
add_subdirectory(tests/e2e/vecsum_while)
add_subdirectory(tests/e2e/vecnorm_l1)
add_subdirectory(tests/e2e/vecnorm_l2)
add_subdirectory(tests/e2e/wildcard_match)
add_subdirectory(tests/e2e/xor_block)
add_subdirectory(tests/unit/gui)
