# Mapper application integration tests.
# Quick tier: small apps on small/medium CGRA templates.
# Full tier: all apps on appropriate templates.
#
# Prerequisites: check-loom-app (produces *.handshake.mlir), check-loom-mapper
#
# Usage:
#   ninja check-loom-mapper-app       # quick tier (smoke tests)
#   ninja check-loom-mapper-app-full  # all tiers

# Template files that must exist for tests to run.
set(MAPPER_APP_TEMPLATES
  "${CMAKE_CURRENT_SOURCE_DIR}/templates/loom_cgra_small.fabric.mlir"
  "${CMAKE_CURRENT_SOURCE_DIR}/templates/loom_cgra_medium.fabric.mlir"
  "${CMAKE_CURRENT_SOURCE_DIR}/templates/loom_cgra_large.fabric.mlir"
)

# Validate that template files are present at configure time.
foreach(tmpl ${MAPPER_APP_TEMPLATES})
  if(NOT EXISTS "${tmpl}")
    message(WARNING "Mapper-app template missing: ${tmpl}")
  endif()
endforeach()

add_custom_target(check-loom-mapper-app
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/mapper_app_test.sh"
    "${LOOM_RUNTIME_DIR}/loom" --tier quick
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  DEPENDS loom
    ${MAPPER_APP_TEMPLATES}
  COMMENT "Mapper integration tests (quick tier)"
)

add_custom_target(check-loom-mapper-app-full
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/mapper_app_test.sh"
    "${LOOM_RUNTIME_DIR}/loom" --tier all
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  DEPENDS loom
    ${MAPPER_APP_TEMPLATES}
  COMMENT "Mapper integration tests (all tiers)"
)
