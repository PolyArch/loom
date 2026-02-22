# Mapper application integration tests.
# Quick tier: small apps on small/medium CGRA templates.
# Full tier: all apps on appropriate templates.
#
# Prerequisites: check-loom-app (produces *.handshake.mlir), check-loom-mapper
#
# Usage:
#   ninja check-loom-mapper-app       # quick tier (smoke tests)
#   ninja check-loom-mapper-app-full  # all tiers

add_custom_target(check-loom-mapper-app
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/mapper_app_test.sh"
    "${LOOM_RUNTIME_DIR}/loom" --tier quick
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  DEPENDS loom
  COMMENT "Mapper integration tests (quick tier)"
)

add_custom_target(check-loom-mapper-app-full
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/mapper_app_test.sh"
    "${LOOM_RUNTIME_DIR}/loom" --tier all
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  DEPENDS loom
  COMMENT "Mapper integration tests (all tiers)"
)
