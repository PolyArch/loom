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

add_executable(loom-accelerator-final-link-test
  AcceleratorFinalLinkTest.cpp
)

# The final-link cases build real relocatable objects and a real archive, so
# they need the host code generator and the pinned archive writer on top of the
# payload library. They take those from the pinned LLVM library this build
# links, because mixing it with the matching static components would register
# the same LLVM command-line options twice.
target_link_libraries(loom-accelerator-final-link-test PRIVATE
  LoomCommon
  LoomFrontendPayload
  LLVM
)
