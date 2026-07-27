add_library(LoomFabricMemoryConsistencyContract
  MemoryConsistencyContract.cpp
)

target_include_directories(LoomFabricMemoryConsistencyContract PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomFabricMemoryConsistencyContract PUBLIC
  LLVM
  LoomFabricIdentity
  LoomFabricResourceContract
)
