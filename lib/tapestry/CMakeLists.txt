add_library(TapestryCore STATIC
  task_graph.cpp
  derived_metrics.cpp
  tdg_emitter.cpp
  compile.cpp
)

target_include_directories(TapestryCore PUBLIC
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

llvm_map_components_to_libnames(TAPESTRY_CORE_LLVM_LIBS
  core support
)

target_link_libraries(TapestryCore PUBLIC
  LoomTDG
  LoomSystemCompiler

  MLIRIR
  MLIRParser
  MLIRSupport

  ${TAPESTRY_CORE_LLVM_LIBS}
)

set_target_properties(TapestryCore PROPERTIES POSITION_INDEPENDENT_CODE ON)

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
