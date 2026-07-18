add_library(LoomFabricBoundaryDataPath
  BoundaryDataPath.cpp
)

target_include_directories(LoomFabricBoundaryDataPath PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

add_dependencies(LoomFabricBoundaryDataPath
  MLIRFabricDialectIncGen
  MLIRFabricEnumsIncGen
  MLIRFabricTypesIncGen
  MLIRFabricOpsIncGen
)

target_link_libraries(LoomFabricBoundaryDataPath PUBLIC
  LLVMSupport
)
