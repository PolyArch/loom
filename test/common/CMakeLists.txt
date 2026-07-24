add_executable(loom-common-artifact-test
  ArtifactTest.cpp
)

target_link_libraries(loom-common-artifact-test PRIVATE
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
