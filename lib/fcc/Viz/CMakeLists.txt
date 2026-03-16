# Generate VizAssets.h from renderer.css and renderer.js at build time
set(VIZ_ASSETS_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(VIZ_ASSETS_OUT ${CMAKE_CURRENT_BINARY_DIR}/VizAssets.h)

add_custom_command(
  OUTPUT ${VIZ_ASSETS_OUT}
  COMMAND ${CMAKE_COMMAND}
    -DCSS_FILE=${VIZ_ASSETS_DIR}/renderer.css
    -DJS_FILE=${VIZ_ASSETS_DIR}/renderer.js
    -DOUT_FILE=${VIZ_ASSETS_OUT}
    -P ${VIZ_ASSETS_DIR}/GenAssets.cmake
  DEPENDS
    ${VIZ_ASSETS_DIR}/renderer.css
    ${VIZ_ASSETS_DIR}/renderer.js
    ${VIZ_ASSETS_DIR}/GenAssets.cmake
  COMMENT "Generating VizAssets.h from CSS/JS"
)

add_custom_target(fcc_viz_assets DEPENDS ${VIZ_ASSETS_OUT})

add_library(FccViz STATIC
  VizExporter.cpp
)

add_dependencies(FccViz fcc_viz_assets)

target_include_directories(FccViz PRIVATE
  ${FCC_INCLUDE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(FccViz PRIVATE
  MLIRIR
  MLIRSupport
)
