add_library(LoomMapping
  Verifier.cpp
)

target_include_directories(LoomMapping PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomMapping PUBLIC
  LLVMSupport
)
