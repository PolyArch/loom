add_executable(loom-common-artifact-test
  ArtifactTest.cpp
)

target_link_libraries(loom-common-artifact-test PRIVATE
  LoomCommon
  LLVMSupport
)

# Route fsync calls in this test binary through the deterministic
# directory-flush failure wrapper defined in ArtifactTest.cpp.
target_link_options(loom-common-artifact-test PRIVATE
  "-Wl,--wrap=fsync"
)

add_executable(loom-common-artifact-local-reference-test
  ArtifactLocalReferenceTest.cpp
)

target_include_directories(loom-common-artifact-local-reference-test PRIVATE
  ${PROJECT_SOURCE_DIR}/lib/Common
)

target_link_libraries(loom-common-artifact-local-reference-test PRIVATE
  LoomCommon
  LLVMSupport
)

add_executable(loom-common-blob-test
  BlobTest.cpp
)

target_link_libraries(loom-common-blob-test PRIVATE
  LoomCommon
  LLVMSupport
)

add_executable(loom-common-canonical-relation-test
  CanonicalRelationTest.cpp
)

target_link_libraries(loom-common-canonical-relation-test PRIVATE
  LoomCommon
  LLVMSupport
)
