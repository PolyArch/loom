# Every payload test takes LLVM from LoomFrontendPayload, which follows the form
# this build's LLVM published. Naming LLVM components again here would mix the
# shared library with its static components in one executable.
add_executable(loom-frontend-payload-framing-test
  FrontendPayloadFramingTest.cpp
)

target_link_libraries(loom-frontend-payload-framing-test PRIVATE
  LoomCommon
  LoomFrontendPayload
)

add_executable(loom-relocatable-payload-test
  RelocatableAcceleratorPayloadTest.cpp
)

target_link_libraries(loom-relocatable-payload-test PRIVATE
  LoomCommon
  LoomFrontendPayload
)

# The hostile-input cases build their canonical bytes through the production
# root codec instead of restating its framing or digest rules.
target_include_directories(loom-relocatable-payload-test PRIVATE
  ${LOOM_SOURCE_DIR}/lib/Frontend/Payload
)

# The final-link cases build real relocatable objects and a real archive, and
# the hand-off case imports the linked module through the same upstream
# LLVM-to-MLIR path the raising driver uses.
add_executable(loom-accelerator-final-link-test
  AcceleratorFinalLinkTest.cpp
)

target_link_libraries(loom-accelerator-final-link-test PRIVATE
  LoomCommon
  LoomFrontendPayload
  MLIRIR
  MLIRLLVMDialect
  MLIRRegisterAllDialects
  MLIRTargetLLVMIRImport
  MLIRFromLLVMIRTranslationRegistration
)
