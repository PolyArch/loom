# LLM Benchmark TaskGraph executables.
#
# Build targets for the three LLM inference TaskGraph definitions.
# These link against TapestryCore for the TaskGraph DSL and MLIR emission.
#
# To build from the main loom build tree, add this to the top-level
# CMakeLists.txt:
#   add_subdirectory(benchmarks/tapestry/ai_llm)

llvm_map_components_to_libnames(LLM_BENCH_LLVM_LIBS core support)

# Helper function for LLM benchmark TaskGraph executables.
function(add_llm_taskgraph_target target_name source_file)
  add_executable(${target_name} ${source_file})

  target_include_directories(${target_name} PRIVATE
    ${LOOM_INCLUDE_DIR}
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
  )

  target_link_libraries(${target_name} PRIVATE
    TapestryCore
    LoomTDG

    MLIRIR
    MLIRParser
    MLIRSupport

    ${LLM_BENCH_LLVM_LIBS}
  )

  set_target_properties(${target_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
  )
endfunction()

# --- TaskGraph executables ---

add_llm_taskgraph_target(llm_taskgraph_decoder taskgraph_decoder.cpp)
add_llm_taskgraph_target(llm_taskgraph_moe     taskgraph_moe.cpp)
add_llm_taskgraph_target(llm_taskgraph_kvcache  taskgraph_kvcache.cpp)
