add_library(LoomHardwareImplementation
  ExternalImplementationContract.cpp
  HardwareImplementation.cpp
  HardwareImplementationValidation.cpp
  ImplementationPayload.cpp
  RepresentationFormat.cpp
  RepresentationLocator.cpp
)

target_include_directories(LoomHardwareImplementation PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomHardwareImplementation PUBLIC
  LoomCommon
  LoomFabricArtifact
  LoomHardwareConfiguration
  LoomImplementationPlatform
  LLVMSupport
)

if(LOOM_HAVE_CIRCT)
  if(NOT TARGET slang)
    message(FATAL_ERROR
      "Pinned CIRCT package does not export the required slang target")
  endif()

  add_library(LoomHardwareRepresentationIndex
    RepresentationIndex.cpp
    RepresentationIndexGate.cpp
    RepresentationIndexRtl.cpp
  )

  target_compile_features(LoomHardwareRepresentationIndex PRIVATE cxx_std_20)
  target_compile_options(LoomHardwareRepresentationIndex PRIVATE
    -fno-exceptions
    -fno-rtti
  )

  target_include_directories(LoomHardwareRepresentationIndex PUBLIC
    $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
  )

  target_link_libraries(LoomHardwareRepresentationIndex PUBLIC
    LoomHardwareImplementation
    LLVMSupport
  PRIVATE
    slang
  )
endif()
