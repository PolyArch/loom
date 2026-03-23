add_library(LoomMultiCoreSim STATIC
  CoreSimWrapper.cpp
  NoCSimModel.cpp
  MemoryHierarchyModel.cpp
  MultiCoreSimSession.cpp
)

target_include_directories(LoomMultiCoreSim PUBLIC
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(LoomMultiCoreSim PUBLIC
  LoomSimulator
  LoomMapper
  MLIRSupport
)
