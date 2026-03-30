set(LOOM_FRONTEND_TEST_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/run_frontend_tests.sh")

add_custom_target(check-loom-frontend
  COMMAND ${LOOM_FRONTEND_TEST_SCRIPT}
  DEPENDS loom "${LOOM_FRONTEND_TEST_SCRIPT}"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Running frontend tests via run_frontend_tests.sh"
  VERBATIM
  USES_TERMINAL
)
