set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Analysis
  CodeGen
  Core
  IPO
  AggressiveInstCombine
  InstCombine
  Instrumentation
  MC
  MCParser
  ObjCARCOpts
  Option
  ScalarOpts
  Support
  TargetParser
  TransformUtils
  Vectorize
)

add_llvm_executable(loom-cc
  loom-cc.cpp
  cc1_main.cpp
  cc1as_main.cpp
  cc1gen_reproducer_main.cpp
  DEPENDS
  intrinsics_gen
  target_parser_gen
)

# Place loom-cc at <build>/bin/ so clang's getMainExecutable resolves the
# resource path to <build>/lib/clang/<major>, matching upstream's layout
# and keeping clang's -print-search-dirs output sane.
set_target_properties(loom-cc PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

llvm_update_compile_flags(loom-cc)

target_link_libraries(loom-cc PRIVATE
  clangBasic
  clangCodeGen
  clangDriver
  clangFrontend
  clangFrontendTool
  clangOptions
  clangSerialization
)

add_dependencies(loom-cc LoomRelocatablePayloadPass)
target_compile_definitions(loom-cc PRIVATE
  LOOM_RELOCATABLE_PAYLOAD_PASS_PATH="$<TARGET_FILE:LoomRelocatablePayloadPass>"
)

mlir_check_all_link_libraries(loom-cc)

# Mirror upstream's clang++/clang-cl symlinks: produce a loom-c++ alongside
# loom-cc so the same binary covers C++ via argv[0] detection.
add_custom_command(TARGET loom-cc POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink
    "$<TARGET_FILE_NAME:loom-cc>"
    "$<TARGET_FILE_DIR:loom-cc>/loom-c++"
  COMMENT "Creating loom-c++ symlink to loom-cc"
  VERBATIM
)

# Reuse the shared LLVM build's clang resource headers (stddef.h, stdarg.h,
# arch intrinsics, etc.). With the binary in <build>/bin/ clang resolves
# the resource path to <build>/lib/clang/<major>; symlinking that to the
# shared LLVM build's prebuilt tree avoids rebuilding the headers locally.
# The link target is absolute, so this layout is suitable for in-tree
# development; a relocatable install rule would copy the directory instead.
set(_LOOM_CC_RESOURCE_LINK "${CMAKE_BINARY_DIR}/lib/clang/${LLVM_VERSION_MAJOR}")
set(_LOOM_CC_RESOURCE_TARGET "${LLVM_LIBRARY_DIR}/clang/${LLVM_VERSION_MAJOR}")
add_custom_command(TARGET loom-cc POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
    "${CMAKE_BINARY_DIR}/lib/clang"
  COMMAND ${CMAKE_COMMAND} -E create_symlink
    "${_LOOM_CC_RESOURCE_TARGET}"
    "${_LOOM_CC_RESOURCE_LINK}"
  COMMENT "Linking loom-cc clang resource directory"
  VERBATIM
)
