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

# An LLVM build may publish one shared library, static components, or both.
# Mixing the two forms in one executable registers the same LLVM command-line
# options twice, so the payload library and everything downstream of it follow
# whichever form that build selected.
if(LLVM_LINK_LLVM_DYLIB AND TARGET LLVM)
  set(loom_payload_llvm_libs LLVM)
else()
  set(loom_payload_llvm_libs
    LLVMBitReader
    LLVMBitWriter
    LLVMCore
    LLVMLTO
    LLVMObject
    LLVMSupport
    LLVMTargetParser
    LLVMTransformUtils
  )
endif()

add_mlir_library(LoomFrontendPayload
  AbiCompatibilityKey.cpp
  AcceleratorFinalLink.cpp
  FrontendConfigView.cpp
  LlvmModuleNormalization.cpp
  PayloadCarrier.cpp
  RelocatableAcceleratorPayload.cpp
  RelocatablePayloadRootCodec.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Frontend/Payload

  LINK_LIBS PUBLIC
  LoomCommon
  ${loom_payload_llvm_libs}
)

add_llvm_pass_plugin(LoomRelocatablePayloadPass
  RelocatablePayloadPass.cpp

  PARTIAL_SOURCES_INTENDED
)

target_link_libraries(LoomRelocatablePayloadPass PRIVATE
  LoomFrontendPayload
)

set_target_properties(LoomRelocatablePayloadPass PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/loom"
)
