# Viz library: self-contained HTML visualization exporter.
#
# Embeds vendored JS/CSS assets as compiled-in C byte arrays so the
# exporter can emit fully self-contained HTML files without runtime I/O.

set(VIZ_ASSET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/assets")
set(VIZ_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${VIZ_GENERATED_DIR}")

# --- Asset embedding via xxd -------------------------------------------

set(VIZ_EMBEDDED_ASSETS
  viz-standalone.js
  d3.min.js
  renderer.js
  renderer.css
)

set(VIZ_ASSET_SOURCES "")

foreach(asset_file ${VIZ_EMBEDDED_ASSETS})
  string(REGEX REPLACE "[^a-zA-Z0-9]" "_" asset_symbol "${asset_file}")
  set(asset_path "${VIZ_ASSET_DIR}/${asset_file}")
  set(generated_c "${VIZ_GENERATED_DIR}/asset_${asset_symbol}.cpp")
  set(generated_h "${VIZ_GENERATED_DIR}/asset_${asset_symbol}.h")

  add_custom_command(
    OUTPUT "${generated_c}" "${generated_h}"
    COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/EmbedAsset.sh"
      "${asset_path}"
      "loom_viz_${asset_symbol}"
      "${generated_c}"
      "${generated_h}"
    DEPENDS "${asset_path}" "${CMAKE_CURRENT_SOURCE_DIR}/EmbedAsset.sh"
    COMMENT "Embedding asset: ${asset_file}"
    VERBATIM
  )

  list(APPEND VIZ_ASSET_SOURCES "${generated_c}")
endforeach()

# --- Library target -----------------------------------------------------

add_llvm_library(LoomViz
  VizHTMLExporter.cpp
  ${VIZ_ASSET_SOURCES}

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    LoomMapperCore
)

target_include_directories(LoomViz
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  PRIVATE
    "${VIZ_GENERATED_DIR}"
    "${LLVM_SOURCE_DIR}/include"
    "${LLVM_BINARY_DIR}/include"
    "${MLIR_SOURCE_DIR}/include"
    "${MLIR_BINARY_DIR}/include"
    "${CIRCT_SOURCE_DIR}/include"
    "${CIRCT_BINARY_DIR}/include"
)

llvm_update_compile_flags(LoomViz)
set_target_properties(LoomViz PROPERTIES POSITION_INDEPENDENT_CODE ON)
