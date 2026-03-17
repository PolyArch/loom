function(fcc_configure_unit_adg_builder target_name)
  target_include_directories(${target_name} PRIVATE
    ${FCC_INCLUDE_DIR}
    ${CMAKE_BINARY_DIR}/include
  )

  llvm_map_components_to_libnames(FCC_UNIT_ADG_LLVM_LIBS
    core support
  )

  target_link_libraries(${target_name} PRIVATE
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

    ${FCC_UNIT_ADG_LLVM_LIBS}
  )

  set_target_properties(${target_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${FCC_RUNTIME_DIR}
  )
endfunction()

function(fcc_add_unit_case unit_name)
  string(MAKE_C_IDENTIFIER "${unit_name}" unit_id)

  set(unit_dir "${CMAKE_CURRENT_SOURCE_DIR}/${unit_name}")
  set(dfg_file "${unit_dir}/dfg.mlir")
  set(output_dir "${CMAKE_SOURCE_DIR}/out/unit/${unit_name}")
  set(stamp_dir "${CMAKE_BINARY_DIR}/tests/unit")
  set(stamp_file "${stamp_dir}/${unit_name}.stamp")
  set(viz_file "${output_dir}/dfg.viz.html")
  set(map_txt_file "${output_dir}/dfg.map.txt")
  set(map_json_file "${output_dir}/dfg.map.json")

  if(NOT EXISTS "${dfg_file}")
    message(FATAL_ERROR "Unit test ${unit_name} is missing dfg.mlir")
  endif()

  if(EXISTS "${unit_dir}/adg.cpp")
    set(builder_target "fcc_unit_adg_${unit_id}")
    set(adg_file "${output_dir}/${unit_name}.fabric.mlir")

    add_executable(${builder_target}
      "${unit_dir}/adg.cpp"
    )
    fcc_configure_unit_adg_builder(${builder_target})

    add_custom_command(
      OUTPUT "${stamp_file}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${stamp_dir}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${output_dir}"
      COMMAND $<TARGET_FILE:${builder_target}> "${adg_file}"
      COMMAND $<TARGET_FILE:fcc> --adg "${adg_file}" --dfg "${dfg_file}" -o "${output_dir}"
      COMMAND ${CMAKE_COMMAND} -E touch "${stamp_file}"
      DEPENDS
        fcc
        ${builder_target}
        "${unit_dir}/adg.cpp"
        "${dfg_file}"
      BYPRODUCTS
        "${adg_file}"
        "${viz_file}"
        "${map_txt_file}"
        "${map_json_file}"
      VERBATIM
    )
  elseif(EXISTS "${unit_dir}/fabric.mlir")
    set(adg_file "${unit_dir}/fabric.mlir")

    add_custom_command(
      OUTPUT "${stamp_file}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${stamp_dir}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${output_dir}"
      COMMAND $<TARGET_FILE:fcc> --adg "${adg_file}" --dfg "${dfg_file}" -o "${output_dir}"
      COMMAND ${CMAKE_COMMAND} -E touch "${stamp_file}"
      DEPENDS
        fcc
        "${adg_file}"
        "${dfg_file}"
      BYPRODUCTS
        "${viz_file}"
        "${map_txt_file}"
        "${map_json_file}"
      VERBATIM
    )
  else()
    message(FATAL_ERROR "Unit test ${unit_name} must provide fabric.mlir or adg.cpp")
  endif()

  set(case_target "fcc-unit-${unit_name}")
  add_custom_target(${case_target} DEPENDS "${stamp_file}")
  set_property(GLOBAL APPEND PROPERTY FCC_UNIT_CASE_TARGETS "${case_target}")
endfunction()

file(GLOB unit_entries RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*")
list(SORT unit_entries)

foreach(unit_name IN LISTS unit_entries)
  if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${unit_name}" AND
     EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${unit_name}/dfg.mlir")
    fcc_add_unit_case("${unit_name}")
  endif()
endforeach()

get_property(fcc_unit_case_targets GLOBAL PROPERTY FCC_UNIT_CASE_TARGETS)

if(fcc_unit_case_targets)
  add_custom_target(check-fcc-unit DEPENDS ${fcc_unit_case_targets})
  if(NOT TARGET check-fcc)
    add_custom_target(check-fcc DEPENDS check-fcc-unit)
  endif()
else()
  add_custom_target(check-fcc-unit)
  if(NOT TARGET check-fcc)
    add_custom_target(check-fcc DEPENDS check-fcc-unit)
  endif()
endif()
