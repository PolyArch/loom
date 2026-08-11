add_library(LoomApplicationPortfolio
  Manifest.cpp
  SourceAdmission.cpp
)

target_include_directories(LoomApplicationPortfolio PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomApplicationPortfolio PUBLIC
  LoomCommon
  LLVMSupport
)
