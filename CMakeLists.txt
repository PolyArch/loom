cmake_minimum_required(VERSION 3.24)
project(loom LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(LOOM_ENABLE_ORTOOLS "Enable OR-Tools backed CP-SAT mapper stages" ON)
option(LOOM_FORCE_INTREE "Force in-tree build even in linked worktrees" OFF)

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

# ---------------------------------------------------------------------------
# Auto-detect linked worktree and reuse pre-built externals from main worktree
# ---------------------------------------------------------------------------
if(NOT LOOM_DEPS_BUILD_DIR AND NOT LOOM_FORCE_INTREE)
  set(_dot_git "${CMAKE_SOURCE_DIR}/.git")
  if(EXISTS "${_dot_git}" AND NOT IS_DIRECTORY "${_dot_git}")
    file(READ "${_dot_git}" _git_content)
    string(STRIP "${_git_content}" _git_content)
    string(REGEX MATCH "gitdir: (.+)" _ "${_git_content}")
    set(_gitdir_raw "${CMAKE_MATCH_1}")
    # Resolve relative paths against CMAKE_SOURCE_DIR
    if(NOT IS_ABSOLUTE "${_gitdir_raw}")
      get_filename_component(_gitdir_raw
        "${CMAKE_SOURCE_DIR}/${_gitdir_raw}" ABSOLUTE)
    endif()
    # Only treat as linked worktree if path matches */.git/worktrees/*
    # (submodules have */.git/modules/* and must NOT trigger this)
    string(FIND "${_gitdir_raw}" "/.git/worktrees/" _wt_pos)
    if(NOT _wt_pos EQUAL -1)
      # Derive main worktree root: <main>/.git/worktrees/<name> -> go up 3
      get_filename_component(_wt_parent "${_gitdir_raw}" DIRECTORY)
      get_filename_component(_git_dir "${_wt_parent}" DIRECTORY)
      get_filename_component(_main_root "${_git_dir}" DIRECTORY)
      set(_main_build "${_main_root}/build")
      set(_has_deps TRUE)
      foreach(_cfg
          "externals/llvm/llvm/lib/cmake/llvm/LLVMConfig.cmake"
          "externals/llvm/llvm/lib/cmake/clang/ClangConfig.cmake"
          "lib/cmake/mlir/MLIRConfig.cmake"
          "lib/cmake/circt/CIRCTConfig.cmake")
        if(NOT EXISTS "${_main_build}/${_cfg}")
          set(_has_deps FALSE)
        endif()
      endforeach()
      if(_has_deps)
        message(STATUS "Linked worktree detected. Reusing externals from: ${_main_build}")
        set(LOOM_DEPS_BUILD_DIR "${_main_build}")
      else()
        message(FATAL_ERROR
          "Linked worktree detected but main worktree build not found or incomplete.\n"
          "Expected build at: ${_main_build}\n"
          "Please build the main worktree first (make rebuild), "
          "or set -DLOOM_FORCE_INTREE=ON to force a full in-tree build.")
      endif()
    endif()
  endif()
endif()

# ---------------------------------------------------------------------------
# Branch: prebuilt-deps mode (linked worktree or explicit LOOM_DEPS_BUILD_DIR)
# ---------------------------------------------------------------------------
if(LOOM_DEPS_BUILD_DIR)
  # Derive package dirs (individual overrides take precedence)
  if(NOT LLVM_DIR)
    set(LLVM_DIR "${LOOM_DEPS_BUILD_DIR}/externals/llvm/llvm/lib/cmake/llvm")
  endif()
  if(NOT Clang_DIR)
    set(Clang_DIR "${LOOM_DEPS_BUILD_DIR}/externals/llvm/llvm/lib/cmake/clang")
  endif()
  if(NOT MLIR_DIR)
    set(MLIR_DIR "${LOOM_DEPS_BUILD_DIR}/lib/cmake/mlir")
  endif()
  if(NOT CIRCT_DIR)
    set(CIRCT_DIR "${LOOM_DEPS_BUILD_DIR}/lib/cmake/circt")
  endif()

  # Find packages (order matters: MLIR before CIRCT)
  find_package(LLVM REQUIRED CONFIG)
  find_package(Clang REQUIRED CONFIG)
  find_package(MLIR REQUIRED CONFIG)
  find_package(CIRCT REQUIRED CONFIG)

  # CMake modules needed for tablegen, add_mlir_dialect_library, etc.
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}" "${MLIR_CMAKE_DIR}")
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)

  # CLANG_INCLUDE_DIRS is already set by find_package(Clang)
  # LLVM_INCLUDE_DIRS, MLIR_INCLUDE_DIRS, CIRCT_INCLUDE_DIRS set by find_package

  # Apply LLVM compile definitions (_GNU_SOURCE, etc.) that HandleLLVMOptions
  # would normally set in an in-tree build.
  separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
  add_definitions(${LLVM_DEFINITIONS_LIST})

  # Derive clang resource directory from the pre-built LLVM
  set(LOOM_RESOURCE_DIR "${LLVM_LIBRARY_DIR}/clang/${LLVM_VERSION_MAJOR}")

  # OR-Tools: still built in-tree (its build-tree exports are not self-contained).
  # We go through externals/CMakeLists.txt which scopes add_compile_options(-w)
  # to the externals/ directory, keeping loom's own warnings intact.
  # externals/CMakeLists.txt skips LLVM when LOOM_DEPS_BUILD_DIR is set.
  add_subdirectory(externals)

# ---------------------------------------------------------------------------
# Branch: in-tree mode (main worktree, default)
# ---------------------------------------------------------------------------
else()
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

  # Build externals (LLVM + CIRCT + OR-Tools)
  add_subdirectory(externals)

  # LLVM/MLIR cmake modules
  list(APPEND CMAKE_MODULE_PATH "${LLVM_BINARY_DIR}/lib/cmake/llvm")
  list(APPEND CMAKE_MODULE_PATH "${MLIR_BINARY_DIR}/lib/cmake/mlir")

  # Derive clang resource directory from the in-tree clang build
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

  # Set package-style variables for uniform downstream use
  set(LLVM_INCLUDE_DIRS
    "${LLVM_SOURCE_DIR}/include" "${LLVM_BINARY_DIR}/include")
  set(MLIR_INCLUDE_DIRS
    "${MLIR_SOURCE_DIR}/include" "${MLIR_BINARY_DIR}/include")
  set(CIRCT_INCLUDE_DIRS
    "${CIRCT_SOURCE_DIR}/include" "${CIRCT_BINARY_DIR}/include")
  set(CLANG_INCLUDE_DIRS
    "${CMAKE_SOURCE_DIR}/externals/llvm/clang/include"
    "${LLVM_BINARY_DIR}/tools/clang/include")
endif()

# ---------------------------------------------------------------------------
# Common configuration (both modes)
# ---------------------------------------------------------------------------

# LOOM include paths
set(LOOM_SOURCE_DIR ${CMAKE_SOURCE_DIR})
set(LOOM_BINARY_DIR ${CMAKE_BINARY_DIR})
set(LOOM_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")

# Runtime output directory
set(LOOM_RUNTIME_DIR "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR})

# LLVM/MLIR/CIRCT includes (package variables set in both branches above)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CIRCT_INCLUDE_DIRS})
include_directories(${LOOM_INCLUDE_DIR})
include_directories(${CMAKE_BINARY_DIR}/include)

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
