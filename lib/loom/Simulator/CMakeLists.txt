add_llvm_library(LoomSimulator
  SimModule.cpp
  SimSwitch.cpp
  SimFifo.cpp
  SimTagOps.cpp
  SimPE.cpp
  SimTemporalPE.cpp
  SimTemporalSW.cpp
  SimMemory.cpp
  SimEngine.cpp
  EventSimSession.cpp

  LINK_LIBS PUBLIC
    LoomMapperCore
    MLIRIR
    MLIRSupport
)

target_include_directories(LoomSimulator
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  PRIVATE
    "${LLVM_SOURCE_DIR}/include"
    "${LLVM_BINARY_DIR}/include"
    "${MLIR_SOURCE_DIR}/include"
    "${MLIR_BINARY_DIR}/include"
    "${CIRCT_SOURCE_DIR}/include"
    "${CIRCT_BINARY_DIR}/include"
)

llvm_update_compile_flags(LoomSimulator)
set_target_properties(LoomSimulator PROPERTIES POSITION_INDEPENDENT_CODE ON)
