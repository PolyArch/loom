add_library(LoomFabricTemporalOperandBuffer
  TemporalOperandBuffer.cpp
  TemporalPeResourceContract.cpp
)

target_include_directories(LoomFabricTemporalOperandBuffer PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

add_dependencies(LoomFabricTemporalOperandBuffer
  MLIRFabricEnumsIncGen
)

target_link_libraries(LoomFabricTemporalOperandBuffer PUBLIC
  LLVM
  LoomFabricIdentity
  LoomFabricResourceContract
)
