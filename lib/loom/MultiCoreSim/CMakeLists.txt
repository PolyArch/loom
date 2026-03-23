add_library(LoomMultiCoreSim STATIC
  InterCoreEvent.cpp
  CoreSimWrapper.cpp
  MultiCoreSimSession.cpp
  FlitModel.cpp
  RouterModel.cpp
  NoCSimModel.cpp
  SPMModel.cpp
  L2BankModel.cpp
  DRAMModel.cpp
  DMAEngine.cpp
  MemoryHierarchyModel.cpp
)

target_include_directories(LoomMultiCoreSim PUBLIC
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(LoomMultiCoreSim PUBLIC
  LoomSimulator
)
