add_library(LoomFabricResourceContract
  ResourceContract.cpp
)

target_include_directories(LoomFabricResourceContract PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomFabricResourceContract PUBLIC
  LLVMSupport
)
