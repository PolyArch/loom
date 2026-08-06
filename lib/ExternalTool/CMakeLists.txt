add_mlir_library(LoomExternalTool
  Binding.cpp
  ExternalFile.cpp
  InvocationBundle.cpp
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
)
