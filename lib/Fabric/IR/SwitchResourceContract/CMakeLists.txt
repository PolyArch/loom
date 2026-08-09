add_library(LoomFabricSwitchResourceContract
  SwitchResourceContract.cpp
)

target_include_directories(LoomFabricSwitchResourceContract PUBLIC
  ${LOOM_SOURCE_DIR}/include
)

target_link_libraries(LoomFabricSwitchResourceContract PUBLIC
  LLVM
  LoomFabricResourceContract
  MLIRFabricEnums
)
