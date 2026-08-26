find_package(OpenSSL REQUIRED COMPONENTS Crypto)

add_mlir_library(LoomExternalTool
  Binding.cpp
  ExternalFile.cpp
  InvocationCommandObservation.cpp
  InvocationCommandSchedule.cpp
  InvocationBundle.cpp
  InvocationCompletion.cpp
  InvocationCache.cpp
  InvocationBundleScript.cpp
  LocalConfig.cpp
  Provider.cpp
  RuntimeBinding.cpp
  ShellProbe.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/ExternalTool

  LINK_LIBS PUBLIC
  LoomCommon
  LLVMSupport
  OpenSSL::Crypto
)
