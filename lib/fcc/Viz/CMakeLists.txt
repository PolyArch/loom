# Generate VizAssets.h from renderer CSS/JS sources and bundled d3 at build time
set(VIZ_ASSETS_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(VIZ_ASSETS_OUT ${CMAKE_CURRENT_BINARY_DIR}/VizAssets.h)
set(VIZ_JS_SOURCES
  ${VIZ_ASSETS_DIR}/renderer/prelude.js
  ${VIZ_ASSETS_DIR}/renderer/minifu.js
  ${VIZ_ASSETS_DIR}/renderer/adg.js
  ${VIZ_ASSETS_DIR}/renderer/dfg.js
  ${VIZ_ASSETS_DIR}/renderer/interaction.js
  ${VIZ_ASSETS_DIR}/renderer/boot.js
)
list(JOIN VIZ_JS_SOURCES ",,FCC_VIZ_SEP,," VIZ_JS_SOURCES_ARG)

add_custom_command(
  OUTPUT ${VIZ_ASSETS_OUT}
  COMMAND ${CMAKE_COMMAND}
    -DCSS_FILE=${VIZ_ASSETS_DIR}/renderer.css
    -DJS_FILES=${VIZ_JS_SOURCES_ARG}
    -DD3_FILE=${VIZ_ASSETS_DIR}/assets/d3.min.js
    -DOUT_FILE=${VIZ_ASSETS_OUT}
    -P ${VIZ_ASSETS_DIR}/GenAssets.cmake
  DEPENDS
    ${VIZ_ASSETS_DIR}/renderer.css
    ${VIZ_JS_SOURCES}
    ${VIZ_ASSETS_DIR}/assets/d3.min.js
    ${VIZ_ASSETS_DIR}/GenAssets.cmake
  COMMENT "Generating VizAssets.h from CSS/JS sources and D3"
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
