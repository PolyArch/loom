add_library(LoomHardwareImplementation
  DefPhysical.cpp
  ExternalImplementationContract.cpp
  FpgaNativeExternalContracts.cpp
  SynopsysDesignWareExternalContract.cpp
  HardwareImplementation.cpp
  HardwareImplementationLocalReference.cpp
  HardwareImplementationValidation.cpp
  ImplementationPayload.cpp
  ImplementationRepresentationRoot.cpp
  PhysicalRepresentationIndex.cpp
  RepresentationFormat.cpp
  RepresentationIndexCore.cpp
  RepresentationIndexDispatch.cpp
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

  target_sources(LoomHardwareImplementation PRIVATE
    RepresentationIndex.cpp
    RepresentationIndexGate.cpp
    RepresentationIndexRtl.cpp
  )

  target_compile_features(LoomHardwareImplementation PRIVATE cxx_std_20)
  target_compile_options(LoomHardwareImplementation PRIVATE
    -fno-exceptions
    -fno-rtti
  )
  target_link_libraries(LoomHardwareImplementation PRIVATE
    slang
  )

  add_library(LoomHardwareRepresentationIndex INTERFACE)
  target_link_libraries(LoomHardwareRepresentationIndex INTERFACE
    LoomHardwareImplementation
  )
else()
  target_sources(LoomHardwareImplementation PRIVATE
    RepresentationIndexUnavailable.cpp
  )
endif()
