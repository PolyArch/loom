if(NOT DEFINED LOOM_BINARY)
  message(FATAL_ERROR "LOOM_BINARY is not set")
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
  if(IS_DIRECTORY "${BUILD_DIR}/tools/loom/CMakeFiles")
    file(REMOVE_RECURSE "${BUILD_DIR}/tools/loom/CMakeFiles")
    message(STATUS "Removed ${BUILD_DIR}/tools/loom/CMakeFiles")
  endif()
endif()
