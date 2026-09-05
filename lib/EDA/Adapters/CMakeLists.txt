add_library(LoomEDAAsicContracts
  AsicStandardCellContracts.cpp
)

target_include_directories(LoomEDAAsicContracts PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomEDAAsicContracts PUBLIC
  LoomHardwareImplementation
  LLVMSupport
)

add_library(LoomEDAPortableGatePublication
  PortableGateImplementation.cpp
)

target_include_directories(LoomEDAPortableGatePublication PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomEDAPortableGatePublication PUBLIC
  LoomEDAAsicContracts
  LoomHardwareRTL
  LLVMSupport
)

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
