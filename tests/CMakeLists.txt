add_subdirectory(mapper)

add_custom_target(check-loom-adg
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/adg_test.sh"
    "${LOOM_RUNTIME_DIR}/loom"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  DEPENDS loom loom-sdk
)

add_custom_target(check-loom-llvm-roundtrip
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/ll_roundtrip.sh"
    "${LOOM_RUNTIME_DIR}/loom" --run
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  DEPENDS loom
)

add_custom_target(check-loom-mlir-roundtrip
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/mlir_roundtrip.sh"
    "${LOOM_RUNTIME_DIR}/loom" --run
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  DEPENDS loom mlir-opt mlir-translate
)

add_custom_target(check-loom-scf-roundtrip
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/scf_roundtrip.sh"
    "${LOOM_RUNTIME_DIR}/loom" --run
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  DEPENDS loom mlir-opt mlir-translate
)

add_custom_target(check-loom-handshake
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/handshake.sh"
    "${LOOM_RUNTIME_DIR}/loom"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  DEPENDS loom
)

add_custom_target(check-loom-handshake-ops-stat
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/handshake_ops_stat.sh"
    "${LOOM_RUNTIME_DIR}/loom"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  DEPENDS loom
)

add_custom_target(check-loom-app
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/ll_roundtrip.sh" "${LOOM_RUNTIME_DIR}/loom" --run
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/mlir_roundtrip.sh" "${LOOM_RUNTIME_DIR}/loom" --run
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/scf_roundtrip.sh" "${LOOM_RUNTIME_DIR}/loom" --run
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/handshake_ops_stat.sh" "${LOOM_RUNTIME_DIR}/loom"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  DEPENDS loom mlir-opt mlir-translate
)

add_custom_target(check-loom-fabric-tdd
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/fabric_tdd.sh"
    "${LOOM_RUNTIME_DIR}/loom"
    "${LLVM_SOURCE_DIR}/utils/lit/lit.py"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  DEPENDS loom FileCheck not
)

add_custom_target(check-loom-fabric
  DEPENDS check-loom-fabric-tdd
)

add_custom_target(check-loom-sv
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/sv_test.sh"
    "${LOOM_RUNTIME_DIR}/loom"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  DEPENDS loom loom-sdk
)

add_custom_target(check-loom-spec-errcode
  COMMAND python3 "${CMAKE_SOURCE_DIR}/tests/spec/error-alignment.py"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

add_custom_target(check-loom-spec-maxloc
  COMMAND python3 "${CMAKE_SOURCE_DIR}/tests/spec/max-loc.py"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

add_custom_target(check-loom-spec
  DEPENDS check-loom-spec-errcode check-loom-spec-maxloc
)

add_custom_target(check-loom
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/check_loom.sh"
    "${LOOM_RUNTIME_DIR}/loom"
    "${LLVM_SOURCE_DIR}/utils/lit/lit.py"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  DEPENDS loom loom-sdk mlir-opt mlir-translate FileCheck not
)

add_custom_target(clean-loom
  COMMAND "${CMAKE_COMMAND}"
    -DLOOM_BINARY="${LOOM_RUNTIME_DIR}/loom"
    -DBUILD_DIR="${CMAKE_BINARY_DIR}"
    -P "${CMAKE_SOURCE_DIR}/tools/loom/clean-loom.cmake"
  COMMAND "${CMAKE_COMMAND}"
    -DSOURCE_DIR="${CMAKE_SOURCE_DIR}"
    -P "${CMAKE_SOURCE_DIR}/tests/cmake/clean-test-output.cmake"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
