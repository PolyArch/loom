add_library(LoomSimulator STATIC
  CycleBackend.cpp
  CycleKernel.cpp
  CycleKernelExecution.cpp
  CycleKernelMemory.cpp
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

target_include_directories(LoomSimulator PUBLIC
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(LoomSimulator PUBLIC
  LoomMapper
  MLIRSupport
)
