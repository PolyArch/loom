add_library(LoomEDAFpgaPublication
  FpgaImplementationPublication.cpp
)

target_include_directories(LoomEDAFpgaPublication PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomEDAFpgaPublication PUBLIC
  LoomHardwareImplementation
  LLVMSupport
)

add_subdirectory(AMD)
add_subdirectory(OpenSource)
add_subdirectory(Synopsys)
add_subdirectory(IntelAltera)
add_subdirectory(Cadence)
