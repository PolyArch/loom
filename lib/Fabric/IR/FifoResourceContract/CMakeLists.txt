add_library(LoomFabricFifoResourceContract
  FifoResourceContract.cpp
)

target_include_directories(LoomFabricFifoResourceContract PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomFabricFifoResourceContract PUBLIC
  LoomFabricResourceContract
)
