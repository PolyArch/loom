cmake_minimum_required(VERSION 3.20)
project(loom LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(LOOM_PNR_INDEX_BITS "32" CACHE STRING
  "Native PnR index width in bits (32 or 64)")
set_property(CACHE LOOM_PNR_INDEX_BITS PROPERTY STRINGS 32 64)
if(NOT LOOM_PNR_INDEX_BITS STREQUAL "32" AND
   NOT LOOM_PNR_INDEX_BITS STREQUAL "64")
  message(FATAL_ERROR
    "LOOM_PNR_INDEX_BITS must be 32 or 64, got '${LOOM_PNR_INDEX_BITS}'")
endif()

file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/include/PnR")
configure_file(
  "${PROJECT_SOURCE_DIR}/include/PnR/BuildConfig.h.in"
  "${PROJECT_BINARY_DIR}/include/PnR/BuildConfig.h"
  @ONLY
)
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# The pinned provider exports monolithic MLIR and LLVM targets alongside its
# component archives. Select one form consistently so local dialect libraries
# cannot pull static definitions into tools that also load the same dylibs.
if(TARGET MLIR AND TARGET LLVM)
  set(MLIR_LINK_MLIR_DYLIB ON)
  set(LLVM_LINK_LLVM_DYLIB ON)
endif()

find_package(Clang REQUIRED CONFIG)
message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")

if(NOT DEFINED ortools_DIR OR ortools_DIR STREQUAL "")
  message(FATAL_ERROR
    "OR-Tools is required; build it with `make or-tools` and configure "
    "through the repository build dispatcher")
endif()
if(NOT DEFINED LOOM_ORTOOLS_SOURCE_COMMIT OR
   LOOM_ORTOOLS_SOURCE_COMMIT STREQUAL "")
  message(FATAL_ERROR
    "LOOM_ORTOOLS_SOURCE_COMMIT is required to validate the exact pinned "
    "OR-Tools package")
endif()
set(_loom_ortools_commit_file
  "${ortools_DIR}/loom-source-commit.txt")
if(NOT EXISTS "${_loom_ortools_commit_file}")
  message(FATAL_ERROR
    "OR-Tools package lacks ${_loom_ortools_commit_file}; refusing an "
    "unvalidated package")
endif()
file(READ "${_loom_ortools_commit_file}" _loom_ortools_source_commit)
string(STRIP "${_loom_ortools_source_commit}"
  _loom_ortools_source_commit)
if(NOT _loom_ortools_source_commit STREQUAL
   LOOM_ORTOOLS_SOURCE_COMMIT)
  message(FATAL_ERROR
    "OR-Tools package source commit ${_loom_ortools_source_commit} does not "
    "match required commit ${LOOM_ORTOOLS_SOURCE_COMMIT}")
endif()
get_filename_component(_loom_ortools_prefix
  "${ortools_DIR}/../../.." ABSOLUTE)
set(_loom_saved_prefix_path "${CMAKE_PREFIX_PATH}")
list(PREPEND CMAKE_PREFIX_PATH "${_loom_ortools_prefix}")
find_package(ortools REQUIRED CONFIG NO_DEFAULT_PATH)
set(CMAKE_PREFIX_PATH "${_loom_saved_prefix_path}")
unset(_loom_saved_prefix_path)
if(NOT TARGET ortools::ortools)
  message(FATAL_ERROR
    "OR-Tools package does not export required target ortools::ortools")
endif()
message(STATUS "Using ortoolsConfig.cmake in: ${ortools_DIR}")

# Optional CIRCT integration. CIRCT is supplied by the shared build under
# externals/circt (see scripts/make-worktree.py, `make circt`). It is
# strictly optional: omit -DCIRCT_DIR to build Loom without CIRCT. A
# provided -DCIRCT_DIR is an exact contract: it must name the directory that
# directly contains CIRCTConfig.cmake, and CIRCT is loaded from precisely that
# directory. A build-root prefix whose CIRCT package is merely nested (e.g.
# <root>/lib/cmake/circt), a directory without a valid CIRCT package, or any
# other CIRCT reachable through CMAKE_PREFIX_PATH or the system search paths is
# a hard error, never a silent skip. An absent -DCIRCT_DIR is reported as an
# explicit status so CIRCT availability is never a mystery.
set(LOOM_HAVE_CIRCT OFF)
if(DEFINED CIRCT_DIR)
  find_package(CIRCT REQUIRED CONFIG NO_DEFAULT_PATH)
  set(LOOM_HAVE_CIRCT ON)
  message(STATUS "Using CIRCTConfig.cmake in: ${CIRCT_DIR}")
else()
  message(STATUS
    "CIRCT unavailable: build it with `make circt`, then reconfigure with "
    "-DCIRCT_DIR=<shared circt cmake dir> to enable CIRCT support")
endif()

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

if(LOOM_HAVE_CIRCT)
  list(APPEND CMAKE_MODULE_PATH "${CIRCT_CMAKE_DIR}")
  include_directories(${CIRCT_INCLUDE_DIRS})
  link_directories(${CIRCT_LIBRARY_DIRS})
endif()

set(LOOM_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(LOOM_BINARY_DIR ${PROJECT_BINARY_DIR})

# The Loom TableGen driver owns the generated operation and implementation
# family registries, so it is defined before any directory that generates from
# them.
add_subdirectory(tools/loom-tblgen)

add_subdirectory(include/Dataflow)
add_subdirectory(include/Fabric)
add_subdirectory(include/Frontend)
add_subdirectory(include/Mapping)
add_subdirectory(lib)
add_subdirectory(tools)

enable_testing()
add_subdirectory(test)
