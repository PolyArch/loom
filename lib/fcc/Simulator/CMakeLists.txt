add_library(FccSimulator STATIC
  FunctionalBackend.cpp
  SimSession.cpp
  SimArtifactWriter.cpp
  SimInputSynthesis.cpp
)

target_include_directories(FccSimulator PUBLIC
  ${FCC_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(FccSimulator PUBLIC
  FccMapper
  MLIRSupport
)
