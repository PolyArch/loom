add_executable(loom-pnr-index-test
  PnrIndexTest.cpp
)

target_link_libraries(loom-pnr-index-test PRIVATE
  LoomPnR
  LLVMSupport
)
