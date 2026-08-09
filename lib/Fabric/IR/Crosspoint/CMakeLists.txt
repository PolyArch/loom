add_library(LoomFabricCrosspoint
  Crosspoint.cpp
)

target_include_directories(LoomFabricCrosspoint PUBLIC
  ${LOOM_SOURCE_DIR}/include
)

target_link_libraries(LoomFabricCrosspoint PUBLIC
  LLVM
)
