find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(LOOM_E2E_RUNNER "${CMAKE_CURRENT_SOURCE_DIR}/run_e2e_tests.py")
set(LOOM_E2E_MANIFEST "${CMAKE_BINARY_DIR}/tests/e2e/e2e_manifest.tsv")
set(LOOM_GEM5_RUNNER "${CMAKE_SOURCE_DIR}/tools/gem5/run_loom_gem5_case.py")

set_property(GLOBAL PROPERTY LOOM_E2E_MANIFEST_LINES "")
set_property(GLOBAL PROPERTY LOOM_E2E_RUNNER_DEPS "")
set_property(GLOBAL PROPERTY LOOM_GEM5_SMOKE_TARGETS "")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/adgs/common
                 ${CMAKE_CURRENT_BINARY_DIR}/adgs_common)

function(loom_configure_e2e_adg_builder target_name)
  target_include_directories(${target_name} PRIVATE
    ${LOOM_INCLUDE_DIR}
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/adgs/common
  )

  llvm_map_components_to_libnames(LOOM_E2E_ADG_LLVM_LIBS
    core support
  )

  target_link_libraries(${target_name} PRIVATE
    LoomE2EADGsCommon
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

    ${LOOM_E2E_ADG_LLVM_LIBS}
  )

  set_target_properties(${target_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
  )
endfunction()

function(loom_add_e2e_adg hw_name)
  string(MAKE_C_IDENTIFIER "${hw_name}" hw_id)
  set(target_name "loom_e2e_adg_${hw_id}")
  set(source_file "${CMAKE_CURRENT_SOURCE_DIR}/adgs/${hw_name}/adg.cpp")
  if(NOT EXISTS "${source_file}")
    message(FATAL_ERROR "Missing e2e ADG builder source: ${source_file}")
  endif()

  add_executable(${target_name} "${source_file}")
  loom_configure_e2e_adg_builder(${target_name})

  set_property(GLOBAL PROPERTY "LOOM_E2E_ADG_TARGET_${hw_name}" "${target_name}")
  set_property(GLOBAL APPEND PROPERTY LOOM_E2E_RUNNER_DEPS ${target_name})
endfunction()

function(loom_add_e2e_pair app_name hw_name)
  set(sim_bundle_template "")
  if(ARGC GREATER 2)
    set(sim_bundle_template "${ARGV2}")
  endif()
  get_property(builder_target GLOBAL PROPERTY "LOOM_E2E_ADG_TARGET_${hw_name}")
  if(NOT builder_target)
    message(FATAL_ERROR "No ADG builder target registered for ${hw_name}")
  endif()

  set(app_dir "${CMAKE_CURRENT_SOURCE_DIR}/apps/${app_name}")
  set(app_source "${app_dir}/${app_name}.cpp")
  if(NOT EXISTS "${app_source}")
    message(FATAL_ERROR "Missing e2e app source: ${app_source}")
  endif()

  set(case_name "${app_name}.${hw_name}")
  set(output_dir "${CMAKE_SOURCE_DIR}/out/e2e/${case_name}")
  set(generated_fabric_file "${output_dir}/${hw_name}.fabric.mlir")

  set(manifest_line
      "${case_name}\t${output_dir}\t${app_source}\t${app_dir}\t$<TARGET_FILE:${builder_target}>\t${generated_fabric_file}\t$<TARGET_FILE:loom>\t${sim_bundle_template}\n")
  set_property(GLOBAL APPEND_STRING PROPERTY LOOM_E2E_MANIFEST_LINES
               "${manifest_line}")
  set_property(GLOBAL APPEND PROPERTY LOOM_E2E_RUNNER_DEPS ${builder_target} loom)

  set(case_target "loom_e2e_${app_name}_${hw_name}")
  add_custom_target(${case_target}
    COMMAND ${Python3_EXECUTABLE} "${LOOM_E2E_RUNNER}"
            --manifest "${LOOM_E2E_MANIFEST}"
            --tests "${case_name}"
    DEPENDS ${builder_target} loom "${LOOM_E2E_MANIFEST}" "${LOOM_E2E_RUNNER}"
    VERBATIM
    USES_TERMINAL
  )
endfunction()

function(loom_add_gem5_smoke case_name)
  set(extra_args "")
  set(target_suffix "")
  if(ARGC GREATER 1)
    set(target_suffix "${ARGV1}")
  endif()
  if(ARGC GREATER 2)
    math(EXPR extra_arg_count "${ARGC} - 2")
    list(SUBLIST ARGV 2 ${extra_arg_count} extra_args)
  endif()
  string(REPLACE "." ";" case_fields "${case_name}")
  list(GET case_fields 0 app_name)
  list(GET case_fields 1 hw_name)
  set(e2e_target "loom_e2e_${app_name}_${hw_name}")
  if(NOT TARGET ${e2e_target})
    message(FATAL_ERROR "No e2e target registered for ${case_name}")
  endif()

  set(output_dir "${CMAKE_SOURCE_DIR}/out/e2e/${case_name}")
  set(gem5_target "loom_gem5_${app_name}_${hw_name}")
  if(target_suffix)
    string(APPEND gem5_target "_${target_suffix}")
  endif()

  add_custom_target(${gem5_target}
    COMMAND ${Python3_EXECUTABLE} "${LOOM_GEM5_RUNNER}"
            --case-dir "${output_dir}"
            ${extra_args}
    DEPENDS ${e2e_target}
            loom
            "${LOOM_GEM5_RUNNER}"
            "${CMAKE_SOURCE_DIR}/configs/loom_cgra.py"
            "${CMAKE_SOURCE_DIR}/runtime/baremetal/crt0.S"
            "${CMAKE_SOURCE_DIR}/runtime/baremetal/loom_baremetal.ld"
    VERBATIM
    USES_TERMINAL
  )

  set_property(GLOBAL APPEND PROPERTY LOOM_GEM5_SMOKE_TARGETS ${gem5_target})
endfunction()

set(LOOM_E2E_ADGS
  mesh-6x6-extmem-1
  mesh-6x6-extmem-2
)

foreach(hw_name IN LISTS LOOM_E2E_ADGS)
  loom_add_e2e_adg("${hw_name}")
endforeach()

set(LOOM_E2E_PAIRS
  "sum-array|mesh-6x6-extmem-1|${CMAKE_CURRENT_SOURCE_DIR}/sim-bundles/sum-array-demo.json"
  "vecadd|mesh-6x6-extmem-2|${CMAKE_CURRENT_SOURCE_DIR}/sim-bundles/vecadd-demo.json"
)

foreach(pair_spec IN LISTS LOOM_E2E_PAIRS)
  string(REPLACE "|" ";" pair_fields "${pair_spec}")
  list(GET pair_fields 0 app_name)
  list(GET pair_fields 1 hw_name)
  list(LENGTH pair_fields pair_len)
  if(pair_len GREATER 2)
    list(GET pair_fields 2 sim_bundle_template)
    loom_add_e2e_pair("${app_name}" "${hw_name}" "${sim_bundle_template}")
  else()
    loom_add_e2e_pair("${app_name}" "${hw_name}")
  endif()
endforeach()

set(LOOM_GEM5_SMOKE_CASES
  "sum-array.mesh-6x6-extmem-1"
  "vecadd.mesh-6x6-extmem-2"
)

foreach(case_name IN LISTS LOOM_GEM5_SMOKE_CASES)
  loom_add_gem5_smoke("${case_name}")
endforeach()

loom_add_gem5_smoke("sum-array.mesh-6x6-extmem-1" "multiaccel" "--extra-accel-count" "1")

get_property(loom_e2e_manifest_lines GLOBAL PROPERTY LOOM_E2E_MANIFEST_LINES)
get_property(loom_e2e_runner_deps GLOBAL PROPERTY LOOM_E2E_RUNNER_DEPS)
get_property(loom_gem5_smoke_targets GLOBAL PROPERTY LOOM_GEM5_SMOKE_TARGETS)
list(REMOVE_DUPLICATES loom_e2e_runner_deps)
list(REMOVE_DUPLICATES loom_gem5_smoke_targets)

file(GENERATE OUTPUT "${LOOM_E2E_MANIFEST}" CONTENT "${loom_e2e_manifest_lines}")

add_custom_target(check-loom-e2e
  COMMAND ${Python3_EXECUTABLE} "${LOOM_E2E_RUNNER}"
          --manifest "${LOOM_E2E_MANIFEST}"
  DEPENDS ${loom_e2e_runner_deps} "${LOOM_E2E_MANIFEST}" "${LOOM_E2E_RUNNER}"
  VERBATIM
  USES_TERMINAL
)

add_custom_target(check-loom-gem5
  DEPENDS ${loom_gem5_smoke_targets}
)
