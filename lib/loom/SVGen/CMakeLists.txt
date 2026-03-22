add_library(LoomSVGen STATIC
  SVGen.cpp
  SVGenFU.cpp
  SVGenFUValidation.cpp
  SVGenPE.cpp
  SVGenTop.cpp
  SVEmitter.cpp
  SVModuleRegistry.cpp
  SVGenConfigLayout.cpp
)

target_include_directories(LoomSVGen PUBLIC
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_include_directories(LoomSVGen PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(LoomSVGen PUBLIC
  LoomFabric
  MLIRSupport
  MLIRIR
)
