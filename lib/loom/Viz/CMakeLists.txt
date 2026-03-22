# Generate VizAssets.h from renderer CSS/JS sources and bundled d3 at build time
set(VIZ_ASSETS_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(VIZ_ASSETS_OUT ${CMAKE_CURRENT_BINARY_DIR}/VizAssets.h)
set(VIZ_JS_SOURCES
  ${VIZ_ASSETS_DIR}/renderer/prelude.js
  ${VIZ_ASSETS_DIR}/renderer/minifu.js
  ${VIZ_ASSETS_DIR}/renderer/adg.js
  ${VIZ_ASSETS_DIR}/renderer/dfg.js
  ${VIZ_ASSETS_DIR}/renderer/interaction.js
  ${VIZ_ASSETS_DIR}/renderer/sim.js
  ${VIZ_ASSETS_DIR}/renderer/boot.js
)
list(JOIN VIZ_JS_SOURCES ",,LOOM_VIZ_SEP,," VIZ_JS_SOURCES_ARG)

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

add_custom_target(loom_viz_assets DEPENDS ${VIZ_ASSETS_OUT})

add_library(LoomViz STATIC
  VizExporter.cpp
  VizExporterADG.cpp
  VizExporterDFG.cpp
  VizExporterLayout.cpp
)

add_dependencies(LoomViz loom_viz_assets)

target_include_directories(LoomViz PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(LoomViz PRIVATE
  MLIRIR
  MLIRSupport
)
