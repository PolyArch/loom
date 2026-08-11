foreach(engine IN ITEMS dfg cgra)
  set(target "loom-gem5-${engine}-engine")
  add_llvm_executable(${target}
    loom-gem5-spatial-engine.cpp
  )
  llvm_update_compile_flags(${target})
  string(TOUPPER "${engine}" engine_upper)
  target_compile_definitions(${target} PRIVATE
    "LOOM_GEM5_SPATIAL_ENGINE_${engine_upper}=1"
  )
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
  target_link_libraries(${target} PRIVATE
    LoomCommon
    LoomMappingArtifact
    LoomRuntimePlatformBinding
    LoomSimulator
    MLIRDataflow
    LLVMSupport
  )
  mlir_check_all_link_libraries(${target})
endforeach()
