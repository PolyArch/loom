add_library(LoomFabricTemporalSwitchResourceContract
  TemporalSwitchResourceContract.cpp
)

target_include_directories(LoomFabricTemporalSwitchResourceContract PUBLIC
  ${LOOM_SOURCE_DIR}/include
)

target_link_libraries(LoomFabricTemporalSwitchResourceContract PUBLIC
  LoomFabricResourceContract
  LLVMSupport
)
