# The payload records the LLVM provider this build selects. Both fields are read
# from the committed submodule declaration so no hand-written constant can drift
# away from the LLVM the build links against.
find_package(Git REQUIRED)

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${LOOM_SOURCE_DIR}/.gitmodules")

execute_process(
  COMMAND ${GIT_EXECUTABLE} ls-tree --object-only HEAD -- externals/llvm
  WORKING_DIRECTORY ${LOOM_SOURCE_DIR}
  OUTPUT_VARIABLE LOOM_LLVM_PROVIDER_COMMIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE loom_llvm_gitlink_result
  ERROR_QUIET
)
string(LENGTH "${LOOM_LLVM_PROVIDER_COMMIT}" loom_llvm_commit_length)
if(NOT loom_llvm_gitlink_result EQUAL 0 OR
   NOT loom_llvm_commit_length EQUAL 40 OR
   NOT LOOM_LLVM_PROVIDER_COMMIT MATCHES "^[0-9a-f]+$")
  message(FATAL_ERROR
    "Could not read the full externals/llvm gitlink commit from "
    "${LOOM_SOURCE_DIR}; the relocatable payload provider cannot be derived.")
endif()

execute_process(
  COMMAND ${GIT_EXECUTABLE} config --file .gitmodules
          --get submodule.externals/llvm.url
  WORKING_DIRECTORY ${LOOM_SOURCE_DIR}
  OUTPUT_VARIABLE loom_llvm_provider_url
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE loom_llvm_url_result
  ERROR_QUIET
)
if(NOT loom_llvm_url_result EQUAL 0 OR loom_llvm_provider_url STREQUAL "")
  message(FATAL_ERROR
    "Could not read the externals/llvm submodule URL from "
    "${LOOM_SOURCE_DIR}/.gitmodules.")
endif()
get_filename_component(loom_llvm_provider_name "${loom_llvm_provider_url}" NAME)
string(REGEX REPLACE "\\.git$" "" LOOM_LLVM_PROVIDER_REPOSITORY
  "${loom_llvm_provider_name}")

file(MAKE_DIRECTORY "${LOOM_BINARY_DIR}/include/Frontend/Payload")
configure_file(
  "${LOOM_SOURCE_DIR}/include/Frontend/Payload/BuildConfig.h.in"
  "${LOOM_BINARY_DIR}/include/Frontend/Payload/BuildConfig.h"
  @ONLY
)

add_mlir_library(LoomFrontendPayload
  AbiCompatibilityKey.cpp
  AcceleratorFinalLink.cpp
  FrontendConfigView.cpp
  LlvmModuleNormalization.cpp
  PayloadCarrier.cpp
  RelocatableAcceleratorPayload.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Frontend/Payload

  LINK_LIBS PUBLIC
  LoomCommon
  LLVMBitReader
  LLVMBitWriter
  LLVMCore
  LLVMLinker
  LLVMObject
  LLVMSupport
  LLVMTargetParser
  LLVMTransformUtils
)
