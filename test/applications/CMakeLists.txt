add_executable(loom-application-manifest-test
  ApplicationManifestTest.cpp
)

target_link_libraries(loom-application-manifest-test PRIVATE
  LoomApplicationPortfolio
  LLVMSupport
)
