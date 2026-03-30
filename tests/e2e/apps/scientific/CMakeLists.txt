set(scicomp_sources
  jacobi_taskgraph.cpp
  cg_taskgraph.cpp
  nbody_taskgraph.cpp
)

foreach(source_file IN LISTS scicomp_sources)
  get_filename_component(target_name "${source_file}" NAME_WE)
  add_executable(${target_name} "${source_file}")

  target_include_directories(${target_name} PRIVATE
    ${LOOM_INCLUDE_DIR}
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
  )

  target_link_libraries(${target_name} PRIVATE
    TapestryCore
  )

  set_target_properties(${target_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
  )

  add_test(NAME ${target_name}
           COMMAND $<TARGET_FILE:${target_name}>)
  set_tests_properties(${target_name} PROPERTIES
    LABELS "scientific;e2e;benchmark"
  )
endforeach()

add_subdirectory(kernels)

add_executable(test_jacobi
  test_jacobi.cpp
  reference/jacobi_ref.cpp
)

target_include_directories(test_jacobi PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)

target_sources(test_jacobi PRIVATE $<TARGET_OBJECTS:scicomp_kernels>)

target_link_libraries(test_jacobi PRIVATE
  TapestryCore
)

set_target_properties(test_jacobi PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)

add_test(NAME test_jacobi
         COMMAND $<TARGET_FILE:test_jacobi>)
set_tests_properties(test_jacobi PROPERTIES
  LABELS "scientific;e2e;benchmark"
)

add_executable(test_cg
  test_cg.cpp
  reference/cg_ref.cpp
)

target_include_directories(test_cg PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)

target_sources(test_cg PRIVATE $<TARGET_OBJECTS:scicomp_kernels>)

target_link_libraries(test_cg PRIVATE
  TapestryCore
)

set_target_properties(test_cg PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)

add_test(NAME test_cg
         COMMAND $<TARGET_FILE:test_cg>)
set_tests_properties(test_cg PROPERTIES
  LABELS "scientific;e2e;benchmark"
)

add_executable(test_nbody
  test_nbody.cpp
  reference/nbody_ref.cpp
)

target_include_directories(test_nbody PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)

target_sources(test_nbody PRIVATE $<TARGET_OBJECTS:scicomp_kernels>)

target_link_libraries(test_nbody PRIVATE
  TapestryCore
)

set_target_properties(test_nbody PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)

add_test(NAME test_nbody
         COMMAND $<TARGET_FILE:test_nbody>)
set_tests_properties(test_nbody PROPERTIES
  LABELS "scientific;e2e;benchmark"
)
