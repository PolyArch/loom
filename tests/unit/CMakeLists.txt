find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(LOOM_UNIT_RUNNER "${CMAKE_CURRENT_SOURCE_DIR}/run_unit_tests.py")
set(LOOM_UNIT_MANIFEST "${CMAKE_BINARY_DIR}/tests/unit/unit_manifest.tsv")

set_property(GLOBAL PROPERTY LOOM_UNIT_MANIFEST_LINES "")
set_property(GLOBAL PROPERTY LOOM_UNIT_RUNNER_DEPS "")

function(loom_configure_unit_adg_builder target_name)
  target_include_directories(${target_name} PRIVATE
    ${LOOM_INCLUDE_DIR}
    ${CMAKE_BINARY_DIR}/include
  )

  llvm_map_components_to_libnames(LOOM_UNIT_ADG_LLVM_LIBS
    core support
  )

  target_link_libraries(${target_name} PRIVATE
    LoomADG
    LoomFabric
    LoomDataflow

    CIRCTHandshake

    MLIRIR
    MLIRParser
    MLIRSupport
    MLIRArithDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRFuncDialect

    ${LOOM_UNIT_ADG_LLVM_LIBS}
  )

  set_target_properties(${target_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
  )
endfunction()

function(loom_add_unit_case unit_name case_name dfg_file)
  string(MAKE_C_IDENTIFIER "${case_name}" unit_id)

  set(unit_dir "${CMAKE_CURRENT_SOURCE_DIR}/${unit_name}")
  set(output_dir "${CMAKE_SOURCE_DIR}/out/unit/${case_name}")
  set(fabric_file "")
  set(builder_target "")
  set(builder_exec "")
  set(generated_fabric_file "")

  if(NOT EXISTS "${dfg_file}")
    message(FATAL_ERROR "Unit test ${case_name} is missing ${dfg_file}")
  endif()

  if(EXISTS "${unit_dir}/adg.cpp")
    set(builder_target "loom_unit_adg_${unit_id}")
    set(generated_fabric_file "${output_dir}/${unit_name}.fabric.mlir")
    set(builder_exec "$<TARGET_FILE:${builder_target}>")

    add_executable(${builder_target}
      "${unit_dir}/adg.cpp"
    )
    loom_configure_unit_adg_builder(${builder_target})
    set_property(GLOBAL APPEND PROPERTY LOOM_UNIT_RUNNER_DEPS ${builder_target})
  elseif(EXISTS "${unit_dir}/fabric.mlir")
    set(fabric_file "${unit_dir}/fabric.mlir")
  else()
    message(FATAL_ERROR "Unit test ${unit_name} must provide fabric.mlir or adg.cpp")
  endif()

  set(manifest_line
      "${case_name}\t${output_dir}\t${dfg_file}\t${fabric_file}\t${builder_exec}\t${generated_fabric_file}\t$<TARGET_FILE:loom>\n")
  set_property(GLOBAL APPEND_STRING PROPERTY LOOM_UNIT_MANIFEST_LINES
               "${manifest_line}")
  set_property(GLOBAL APPEND PROPERTY LOOM_UNIT_RUNNER_DEPS loom)

  set(case_target "loom-unit-${case_name}")
  set(case_deps loom "${LOOM_UNIT_MANIFEST}" "${LOOM_UNIT_RUNNER}")
  if(builder_target)
    list(APPEND case_deps ${builder_target})
  endif()

  add_custom_target(${case_target}
    COMMAND ${Python3_EXECUTABLE} "${LOOM_UNIT_RUNNER}"
            --manifest "${LOOM_UNIT_MANIFEST}"
            --tests "${case_name}"
    DEPENDS ${case_deps}
    VERBATIM
    USES_TERMINAL
  )
endfunction()

file(GLOB unit_entries CONFIGURE_DEPENDS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "*"
)
list(SORT unit_entries)

foreach(unit_name IN LISTS unit_entries)
  if(NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${unit_name}")
    continue()
  endif()

  set(unit_dir "${CMAKE_CURRENT_SOURCE_DIR}/${unit_name}")
  if(EXISTS "${unit_dir}/dfg.mlir")
    loom_add_unit_case("${unit_name}" "${unit_name}" "${unit_dir}/dfg.mlir")
  endif()

  file(GLOB extra_dfgs CONFIGURE_DEPENDS
    "${unit_dir}/dfg-*.mlir"
  )
  list(SORT extra_dfgs)
  foreach(extra_dfg IN LISTS extra_dfgs)
    get_filename_component(extra_dfg_name "${extra_dfg}" NAME)
    string(REGEX REPLACE "^dfg-(.*)\\.mlir$" "\\1" case_suffix
           "${extra_dfg_name}")
    set(case_name "${unit_name}-${case_suffix}")
    loom_add_unit_case("${unit_name}" "${case_name}" "${extra_dfg}")
  endforeach()
endforeach()

get_property(loom_unit_manifest_lines GLOBAL PROPERTY LOOM_UNIT_MANIFEST_LINES)
get_property(loom_unit_runner_deps GLOBAL PROPERTY LOOM_UNIT_RUNNER_DEPS)
list(REMOVE_DUPLICATES loom_unit_runner_deps)

file(GENERATE OUTPUT "${LOOM_UNIT_MANIFEST}" CONTENT "${loom_unit_manifest_lines}")

add_custom_target(check-loom-unit
  COMMAND ${Python3_EXECUTABLE} "${LOOM_UNIT_RUNNER}"
          --manifest "${LOOM_UNIT_MANIFEST}"
  DEPENDS ${loom_unit_runner_deps} "${LOOM_UNIT_MANIFEST}" "${LOOM_UNIT_RUNNER}"
  VERBATIM
  USES_TERMINAL
)

if(NOT TARGET check-loom)
  add_custom_target(check-loom DEPENDS check-loom-unit)
endif()

# Tapestry (TDG dialect) C++ tests
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tapestry")
  add_subdirectory(tapestry)
endif()

# Fabric dialect C++ tests
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/fabric")
  add_subdirectory(fabric)
endif()
