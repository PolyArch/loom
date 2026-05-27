cmake_minimum_required(VERSION 3.20)
project(loom LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(LOOM_ENABLE_ILP "Enable HiGHS-backed ILP partitioner" ON)
if(LOOM_ENABLE_ILP)
  find_package(HIGHS CONFIG QUIET)
  if(NOT TARGET highs::highs)
    include(FetchContent)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(FAST_BUILD ON CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(USE_OPENMP OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(highs
      GIT_REPOSITORY https://github.com/ERGO-Code/HiGHS
      GIT_TAG v1.14.0)
    FetchContent_MakeAvailable(highs)
  endif()
  add_compile_definitions(LOOM_HAS_ILP)
endif()

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

find_package(Clang REQUIRED CONFIG)
message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")

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

set(LOOM_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(LOOM_BINARY_DIR ${PROJECT_BINARY_DIR})

add_subdirectory(include/Fabric)
add_subdirectory(include/Dataflow)
add_subdirectory(lib)
add_subdirectory(tools)

enable_testing()
add_subdirectory(test)
