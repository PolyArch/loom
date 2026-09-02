add_library(LoomHardwareImplementation
  DefPhysical.cpp
  ExternalImplementationContract.cpp
  FabricModel.cpp
  FpgaNativeExternalContracts.cpp
  SynopsysDesignWareExternalContract.cpp
  HardwareImplementation.cpp
  HardwareImplementationDiagnostics.cpp
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
  # slang is compiled without exceptions, so Boost requires the program to
  # define boost::throw_exception. The definition lives in its own archive that
  # the linker rescans together with slang: whichever slang member is
  # extracted, including one pulled in for a shared template instantiation,
  # resolves it regardless of archive order.
  add_library(LoomSlangExceptionBoundary STATIC SlangExceptionBoundary.cpp)
  target_link_libraries(LoomHardwareImplementation PRIVATE
    "$<LINK_GROUP:RESCAN,slang,LoomSlangExceptionBoundary>"
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
