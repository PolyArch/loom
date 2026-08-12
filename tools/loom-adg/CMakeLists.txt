add_llvm_executable(loom-adg
  loom-adg.cpp
)

llvm_update_compile_flags(loom-adg)

target_link_libraries(loom-adg PRIVATE
  LoomADGBuilder
  LoomCommon
  LoomConfig
  LLVMSupport
)
