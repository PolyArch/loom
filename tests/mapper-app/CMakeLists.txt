# Mapper application integration tests.
# Quick tier: small apps on small/medium CGRA templates.
# Full tier: all apps on appropriate templates.
#
# Prerequisites: check-loom-app (produces *.handshake.mlir), check-loom-mapper
#
# Usage:
#   ninja check-loom-mapper-app       # quick tier (smoke tests)
#   ninja check-loom-mapper-app-full  # all tiers

# Template generator sources and outputs.
set(TEMPLATES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/templates")
set(TEMPLATES_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/templates")
set(LOOM_BIN "${LOOM_RUNTIME_DIR}/loom")

set(MAPPER_APP_TEMPLATES)
set(MAPPER_APP_GEN_TARGETS)

foreach(tier smoke small medium large)
  set(GEN_SRC "${TEMPLATES_DIR}/gen_${tier}.cpp")
  set(GEN_BIN "${TEMPLATES_BIN_DIR}/gen_${tier}")
  set(GEN_OUT "${TEMPLATES_DIR}/loom_cgra_${tier}.fabric.mlir")

  # Compile the generator using loom --as-clang.
  add_custom_command(
    OUTPUT "${GEN_BIN}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${TEMPLATES_BIN_DIR}"
    COMMAND "${LOOM_BIN}" --as-clang "${GEN_SRC}" -o "${GEN_BIN}"
    DEPENDS loom "${GEN_SRC}" "${TEMPLATES_DIR}/cgra_pe_catalog.h"
    COMMENT "Compiling ${tier} template generator"
  )

  # Run the generator to produce the .fabric.mlir template.
  add_custom_command(
    OUTPUT "${GEN_OUT}"
    COMMAND "${GEN_BIN}" "${GEN_OUT}"
    DEPENDS "${GEN_BIN}"
    COMMENT "Generating loom_cgra_${tier}.fabric.mlir"
  )

  list(APPEND MAPPER_APP_TEMPLATES "${GEN_OUT}")
  list(APPEND MAPPER_APP_GEN_TARGETS "${GEN_OUT}")
endforeach()

# Convenience target to generate all templates.
add_custom_target(loom-mapper-app-templates
  DEPENDS ${MAPPER_APP_GEN_TARGETS}
  COMMENT "Generating all CGRA templates"
)

