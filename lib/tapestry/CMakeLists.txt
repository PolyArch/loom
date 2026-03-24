add_library(TapestryOptimizer STATIC
  tdg_optimizer.cpp
  host_scheduler.cpp
)

target_include_directories(TapestryOptimizer PUBLIC
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(TapestryOptimizer PUBLIC
  LoomSystemCompiler

  MLIRIR
  MLIRSupport
)

set_target_properties(TapestryOptimizer PROPERTIES POSITION_INDEPENDENT_CODE ON)
