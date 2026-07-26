add_library(LoomFabricMemoryConnectivityContract
  MemoryConnectivityContract.cpp
  MemoryConnectivityContractCodec.cpp
)

target_include_directories(LoomFabricMemoryConnectivityContract PUBLIC
  ${LOOM_SOURCE_DIR}/include
)

target_link_libraries(LoomFabricMemoryConnectivityContract PUBLIC
  LoomFabricMemoryOperationPort
  LLVMSupport
)
