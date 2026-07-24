add_executable(loom-frontend-payload-framing-test
  FrontendPayloadFramingTest.cpp
)

target_link_libraries(loom-frontend-payload-framing-test PRIVATE
  LoomCommon
  LoomFrontendPayload
  LLVMSupport
)

add_executable(loom-relocatable-payload-test
  RelocatableAcceleratorPayloadTest.cpp
)

target_link_libraries(loom-relocatable-payload-test PRIVATE
  LoomCommon
  LoomFrontendPayload
  LLVMAsmParser
  LLVMBitReader
  LLVMBitWriter
  LLVMCore
  LLVMSupport
  LLVMTargetParser
)
