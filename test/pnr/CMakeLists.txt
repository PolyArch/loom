add_executable(loom-pnr-index-test
  PnrIndexTest.cpp
)

target_link_libraries(loom-pnr-index-test PRIVATE
  LoomPnR
  LLVMSupport
)

add_executable(loom-pnr-route-tree-state-test
  RouteTreeStateTest.cpp
  ../mapping/MappingCoreTestSupport.cpp
)

target_include_directories(loom-pnr-route-tree-state-test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../mapping
)

target_link_libraries(loom-pnr-route-tree-state-test PRIVATE
  LoomMapping
  LoomPnR
  LoomFabricBoundaryDataPath
  MLIRIR
  LLVMSupport
)
