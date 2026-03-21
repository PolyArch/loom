add_library(FccSVGen STATIC
  SVGen.cpp
  SVGenFU.cpp
  SVGenPE.cpp
  SVGenTop.cpp
  SVEmitter.cpp
  SVModuleRegistry.cpp
  SVGenConfigLayout.cpp
)

target_include_directories(FccSVGen PUBLIC
  ${FCC_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_include_directories(FccSVGen PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(FccSVGen PUBLIC
  FccFabric
  MLIRSupport
  MLIRIR
)
