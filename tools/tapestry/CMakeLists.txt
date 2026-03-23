# Common libraries shared by all tapestry tools
set(TAPESTRY_COMMON_LIBS
  LoomSystemCompiler
  LoomMultiCoreSim
  LoomSVGen
  LoomMapper
  LoomSimulator
  LoomADG
  LoomDataflow
  LoomFabric

  CIRCTHandshake

  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRIR
  MLIRParser
  MLIRSupport
)

llvm_map_components_to_libnames(TAPESTRY_LLVM_LIBS
  core support
)

# tapestry-compile: multi-core compilation only
add_executable(tapestry-compile tapestry_compile.cpp)
target_include_directories(tapestry-compile PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)
target_link_libraries(tapestry-compile PRIVATE
  ${TAPESTRY_COMMON_LIBS}
  ${TAPESTRY_LLVM_LIBS}
)
set_target_properties(tapestry-compile PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)

# tapestry-simulate: compilation + simulation
add_executable(tapestry-simulate tapestry_simulate.cpp)
target_include_directories(tapestry-simulate PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)
target_link_libraries(tapestry-simulate PRIVATE
  ${TAPESTRY_COMMON_LIBS}
  ${TAPESTRY_LLVM_LIBS}
)
set_target_properties(tapestry-simulate PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)

# tapestry-rtlgen: compilation + RTL generation
add_executable(tapestry-rtlgen tapestry_rtlgen.cpp)
target_include_directories(tapestry-rtlgen PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)
target_link_libraries(tapestry-rtlgen PRIVATE
  ${TAPESTRY_COMMON_LIBS}
  ${TAPESTRY_LLVM_LIBS}
)
set_target_properties(tapestry-rtlgen PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)

# tapestry-pipeline: full end-to-end pipeline
add_executable(tapestry-pipeline tapestry_pipeline.cpp)
target_include_directories(tapestry-pipeline PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)
target_link_libraries(tapestry-pipeline PRIVATE
  ${TAPESTRY_COMMON_LIBS}
  ${TAPESTRY_LLVM_LIBS}
)
set_target_properties(tapestry-pipeline PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)
