if(NOT DEFINED SOURCE_DIR)
  message(FATAL_ERROR "SOURCE_DIR is not set")
endif()

# Remove app test output directories
file(GLOB APP_OUTPUT_DIRS "${SOURCE_DIR}/tests/app/*/Output")
foreach(dir IN LISTS APP_OUTPUT_DIRS)
  if(IS_DIRECTORY "${dir}")
    file(REMOVE_RECURSE "${dir}")
    message(STATUS "Removed ${dir}")
  endif()
endforeach()

# Remove generated parallel files
file(GLOB APP_PARALLEL_FILES "${SOURCE_DIR}/tests/app/*.parallel.sh")
foreach(f IN LISTS APP_PARALLEL_FILES)
  file(REMOVE "${f}")
  message(STATUS "Removed ${f}")
endforeach()

# Remove ops stat files
file(GLOB OPS_STAT_FILES "${SOURCE_DIR}/tests/app/full-ops-*.stat")
foreach(f IN LISTS OPS_STAT_FILES)
  file(REMOVE "${f}")
  message(STATUS "Removed ${f}")
endforeach()

# Remove ADG test output directories
file(GLOB ADG_OUTPUT_DIRS "${SOURCE_DIR}/tests/adg/*/Output")
foreach(dir IN LISTS ADG_OUTPUT_DIRS)
  if(IS_DIRECTORY "${dir}")
    file(REMOVE_RECURSE "${dir}")
    message(STATUS "Removed ${dir}")
  endif()
endforeach()

# Remove ADG parallel files
file(GLOB ADG_PARALLEL_FILES "${SOURCE_DIR}/tests/adg/*.parallel.sh")
foreach(f IN LISTS ADG_PARALLEL_FILES)
  file(REMOVE "${f}")
  message(STATUS "Removed ${f}")
endforeach()

# Remove SV test output directories
file(GLOB SV_OUTPUT_DIRS "${SOURCE_DIR}/tests/sv/*/Output")
foreach(dir IN LISTS SV_OUTPUT_DIRS)
  if(IS_DIRECTORY "${dir}")
    file(REMOVE_RECURSE "${dir}")
    message(STATUS "Removed ${dir}")
  endif()
endforeach()

# Remove SV parallel files
file(GLOB SV_PARALLEL_FILES "${SOURCE_DIR}/tests/sv/*.parallel.sh")
foreach(f IN LISTS SV_PARALLEL_FILES)
  file(REMOVE "${f}")
  message(STATUS "Removed ${f}")
endforeach()

# Remove fabric TDD output directories
file(GLOB TDD_OUTPUT_DIRS "${SOURCE_DIR}/tests/fabric/tdd/*/Output")
foreach(dir IN LISTS TDD_OUTPUT_DIRS)
  if(IS_DIRECTORY "${dir}")
    file(REMOVE_RECURSE "${dir}")
    message(STATUS "Removed ${dir}")
  endif()
endforeach()

# Remove results directory
if(IS_DIRECTORY "${SOURCE_DIR}/tests/.results")
  file(REMOVE_RECURSE "${SOURCE_DIR}/tests/.results")
  message(STATUS "Removed ${SOURCE_DIR}/tests/.results")
endif()
