add_library(LoomFabricResourceContract
  OperationResourceContract.cpp
  PhysicalTagResourceContract.cpp
  ResourceContract.cpp
  ResourceContractRecord.cpp
  UsePatternValue.cpp
)

target_include_directories(LoomFabricResourceContract PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomFabricResourceContract PUBLIC
  LLVM
  LoomFabricPhysicalTag
)
