add_library(LoomMapping
  StructureValidator.cpp
)

target_include_directories(LoomMapping PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomMapping PUBLIC
  LLVMSupport
)
