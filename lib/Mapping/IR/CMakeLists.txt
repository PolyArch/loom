add_mlir_dialect_library(MLIRMapping
  MappingAttrs.cpp
  MappingDialect.cpp
  MappingEnums.cpp
  MappingOps.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Mapping

  DEPENDS
  MLIRMappingAttrsIncGen
  MLIRMappingDialectIncGen
  MLIRMappingEnumsIncGen
  MLIRMappingOpsIncGen
)

target_link_libraries(MLIRMapping PUBLIC
  LoomCommon
  LoomFabricIdentity
  MLIRDataflow
)

mlir_target_link_libraries(MLIRMapping PUBLIC
  MLIRIR
  MLIRSupport
)
