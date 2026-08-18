add_library(LoomPnRDerivedContextSession
  ../PnrDerivedContextSession.cpp
)

target_include_directories(LoomPnRDerivedContextSession PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomPnRDerivedContextSession PUBLIC
  LoomCommon
  LLVMSupport
)
