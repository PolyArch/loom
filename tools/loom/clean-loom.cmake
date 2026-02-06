if(NOT DEFINED LOOM_BINARY)
  message(FATAL_ERROR "LOOM_BINARY is not set")
endif()
if(NOT DEFINED SOURCE_DIR)
  message(FATAL_ERROR "SOURCE_DIR is not set")
endif()

# Remove loom binary
if(EXISTS "${LOOM_BINARY}")
  file(REMOVE "${LOOM_BINARY}")
  message(STATUS "Removed ${LOOM_BINARY}")
endif()

# Remove loom library and tool build artifacts
if(DEFINED BUILD_DIR)
  if(IS_DIRECTORY "${BUILD_DIR}/lib/loom")
    file(REMOVE_RECURSE "${BUILD_DIR}/lib/loom")
    message(STATUS "Removed ${BUILD_DIR}/lib/loom")
  endif()
  # Remove tool object files but not cmake scripts
  if(IS_DIRECTORY "${BUILD_DIR}/tools/loom/CMakeFiles")
    file(REMOVE_RECURSE "${BUILD_DIR}/tools/loom/CMakeFiles")
    message(STATUS "Removed ${BUILD_DIR}/tools/loom/CMakeFiles")
  endif()
endif()

# Remove test output directories
file(GLOB APP_OUTPUT_DIRS "${SOURCE_DIR}/tests/app/*/Output")
foreach(dir IN LISTS APP_OUTPUT_DIRS)
  if(IS_DIRECTORY "${dir}")
    file(REMOVE_RECURSE "${dir}")
    message(STATUS "Removed ${dir}")
  endif()
endforeach()
