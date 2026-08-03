add_library(LoomFabricPhysicalTag
  PhysicalTag.cpp
)

target_include_directories(LoomFabricPhysicalTag PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomFabricPhysicalTag PUBLIC
  LLVMSupport
)
