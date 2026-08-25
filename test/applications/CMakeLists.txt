add_executable(loom-application-manifest-test
  ApplicationManifestTest.cpp
)

add_executable(loom-application-host-run-test
  ApplicationHostRunnerTest.cpp
)

get_filename_component(LOOM_TEST_REPOSITORY_ROOT
  "${LOOM_EXTERNAL_SOURCE_DIR}" DIRECTORY)

target_compile_definitions(loom-application-manifest-test PRIVATE
  LOOM_TEST_REPOSITORY_ROOT="${LOOM_TEST_REPOSITORY_ROOT}"
)

target_compile_definitions(loom-application-host-run-test PRIVATE
  LOOM_TEST_REPOSITORY_ROOT="${LOOM_TEST_REPOSITORY_ROOT}"
)

target_link_libraries(loom-application-manifest-test PRIVATE
  LoomApplicationPortfolio
  LLVMSupport
)

target_link_libraries(loom-application-host-run-test PRIVATE
  LoomApplicationHostRunner
  LLVMSupport
)
