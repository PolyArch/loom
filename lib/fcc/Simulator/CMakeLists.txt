add_library(FccSimulator STATIC
  CycleBackend.cpp
  CycleKernel.cpp
  FunctionalBackend.cpp
  FunctionalBackendOps.cpp
  RuntimeImageBuilder.cpp
  RuntimeImageBinary.cpp
  SimTypes.cpp
  SimStats.cpp
  SimFunctionUnit.cpp
  SimFunctionUnitCore.cpp
  SimFunctionUnitStateful.cpp
  SimTemporalPE.cpp
  SimMemory.cpp
  SimModule.cpp
  RuntimeImage.cpp
  StaticModel.cpp
  StaticModelBuilder.cpp
  SimBundle.cpp
  SimSession.cpp
  SimArtifactWriter.cpp
  SimInputSynthesis.cpp
  PortTraceExporter.cpp
)

target_include_directories(FccSimulator PUBLIC
  ${FCC_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(FccSimulator PUBLIC
  FccMapper
  MLIRSupport
)
