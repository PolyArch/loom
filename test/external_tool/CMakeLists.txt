add_executable(loom-external-tool-local-config-test
  LocalConfigTest.cpp
)

target_link_libraries(loom-external-tool-local-config-test PRIVATE
  LoomExternalTool
  LLVMSupport
)

add_executable(loom-external-tool-external-file-test
  ExternalFileTest.cpp
)

target_link_libraries(loom-external-tool-external-file-test PRIVATE
  LoomExternalTool
  LLVMSupport
)

add_executable(loom-external-tool-binding-test
  BindingTest.cpp
)

target_link_libraries(loom-external-tool-binding-test PRIVATE
  LoomExternalTool
  LLVMSupport
)

add_executable(loom-external-tool-shell-probe-test
  ShellProbeTest.cpp
)

target_link_libraries(loom-external-tool-shell-probe-test PRIVATE
  LoomExternalTool
  LLVMSupport
)

add_executable(loom-external-tool-invocation-bundle-test
  InvocationBundleTest.cpp
)

target_link_libraries(loom-external-tool-invocation-bundle-test PRIVATE
  LoomExternalTool
  LLVMSupport
)

add_executable(loom-external-tool-invocation-receipt-test
  InvocationReceiptTest.cpp
)

target_link_libraries(loom-external-tool-invocation-receipt-test PRIVATE
  LoomExternalTool
  LLVMSupport
)

add_executable(loom-external-tool-command-schedule-test
  InvocationCommandScheduleTest.cpp
)

target_link_libraries(loom-external-tool-command-schedule-test PRIVATE
  LoomExternalTool
  LLVMSupport
)

add_executable(loom-external-tool-runtime-binding-test
  RuntimeBindingTest.cpp
)

target_link_libraries(loom-external-tool-runtime-binding-test PRIVATE
  LoomExternalTool
  LLVMSupport
)

add_executable(loom-external-tool-provider-test
  ProviderTest.cpp
)

target_link_libraries(loom-external-tool-provider-test PRIVATE
  LoomExternalTool
  LLVMSupport
)
