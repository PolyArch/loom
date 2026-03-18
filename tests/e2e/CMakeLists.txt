find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(FCC_E2E_RUNNER "${CMAKE_CURRENT_SOURCE_DIR}/run_e2e_tests.py")
set(FCC_E2E_MANIFEST "${CMAKE_BINARY_DIR}/tests/e2e/e2e_manifest.tsv")

set_property(GLOBAL PROPERTY FCC_E2E_MANIFEST_LINES "")
set_property(GLOBAL PROPERTY FCC_E2E_RUNNER_DEPS "")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/adgs/common
                 ${CMAKE_CURRENT_BINARY_DIR}/adgs_common)

function(fcc_configure_e2e_adg_builder target_name)
  target_include_directories(${target_name} PRIVATE
    ${FCC_INCLUDE_DIR}
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/adgs/common
  )

  llvm_map_components_to_libnames(FCC_E2E_ADG_LLVM_LIBS
    core support
  )

  target_link_libraries(${target_name} PRIVATE
    FccE2EADGsCommon
    FccADG
    FccFabric
    FccDataflow

    CIRCTHandshake

    MLIRIR
    MLIRParser
    MLIRSupport
    MLIRArithDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRFuncDialect

    ${FCC_E2E_ADG_LLVM_LIBS}
  )

  set_target_properties(${target_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${FCC_RUNTIME_DIR}
  )
endfunction()

function(fcc_add_e2e_adg hw_name)
  string(MAKE_C_IDENTIFIER "${hw_name}" hw_id)
  set(target_name "fcc_e2e_adg_${hw_id}")
  set(source_file "${CMAKE_CURRENT_SOURCE_DIR}/adgs/${hw_name}/adg.cpp")
  if(NOT EXISTS "${source_file}")
    message(FATAL_ERROR "Missing e2e ADG builder source: ${source_file}")
  endif()

  add_executable(${target_name} "${source_file}")
  fcc_configure_e2e_adg_builder(${target_name})

  set_property(GLOBAL PROPERTY "FCC_E2E_ADG_TARGET_${hw_name}" "${target_name}")
  set_property(GLOBAL APPEND PROPERTY FCC_E2E_RUNNER_DEPS ${target_name})
endfunction()

function(fcc_add_e2e_pair app_name hw_name)
  get_property(builder_target GLOBAL PROPERTY "FCC_E2E_ADG_TARGET_${hw_name}")
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
      "${case_name}\t${output_dir}\t${app_source}\t${app_dir}\t$<TARGET_FILE:${builder_target}>\t${generated_fabric_file}\t$<TARGET_FILE:fcc>\n")
  set_property(GLOBAL APPEND_STRING PROPERTY FCC_E2E_MANIFEST_LINES
               "${manifest_line}")
  set_property(GLOBAL APPEND PROPERTY FCC_E2E_RUNNER_DEPS ${builder_target} fcc)

  string(MAKE_C_IDENTIFIER "${case_name}" case_id)
  set(case_target "fcc-e2e-${case_id}")
  add_custom_target(${case_target}
    COMMAND ${Python3_EXECUTABLE} "${FCC_E2E_RUNNER}"
            --manifest "${FCC_E2E_MANIFEST}"
            --tests "${case_name}"
    DEPENDS ${builder_target} fcc "${FCC_E2E_MANIFEST}" "${FCC_E2E_RUNNER}"
    VERBATIM
    USES_TERMINAL
  )
endfunction()

set(FCC_E2E_ADGS
  tiny-star-4pe
  medium-star-8pe
  wide-star-16pe
  medium-chess-6x6
  medium-chess-10x10
)

foreach(hw_name IN LISTS FCC_E2E_ADGS)
  fcc_add_e2e_adg("${hw_name}")
endforeach()

set(FCC_E2E_PAIRS
  "vecadd|tiny-star-4pe"
  "dotprod|tiny-star-4pe"
  "gemv|medium-star-8pe"
  "spmv|medium-star-8pe"
  "gemm|wide-star-16pe"
  "gemv|wide-star-16pe"
  "gemm|medium-chess-6x6"
  "merge-sort|medium-chess-6x6"
  "spmm|medium-chess-10x10"
  "bfs-level|medium-chess-10x10"
)

foreach(pair_spec IN LISTS FCC_E2E_PAIRS)
  string(REPLACE "|" ";" pair_fields "${pair_spec}")
  list(GET pair_fields 0 app_name)
  list(GET pair_fields 1 hw_name)
  fcc_add_e2e_pair("${app_name}" "${hw_name}")
endforeach()

get_property(fcc_e2e_manifest_lines GLOBAL PROPERTY FCC_E2E_MANIFEST_LINES)
get_property(fcc_e2e_runner_deps GLOBAL PROPERTY FCC_E2E_RUNNER_DEPS)
list(REMOVE_DUPLICATES fcc_e2e_runner_deps)

file(GENERATE OUTPUT "${FCC_E2E_MANIFEST}" CONTENT "${fcc_e2e_manifest_lines}")

add_custom_target(check-fcc-e2e
  COMMAND ${Python3_EXECUTABLE} "${FCC_E2E_RUNNER}"
          --manifest "${FCC_E2E_MANIFEST}"
  DEPENDS ${fcc_e2e_runner_deps} "${FCC_E2E_MANIFEST}" "${FCC_E2E_RUNNER}"
  VERBATIM
  USES_TERMINAL
)
