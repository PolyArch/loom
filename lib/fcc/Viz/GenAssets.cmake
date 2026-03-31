# GenAssets.cmake - Generate VizAssets.h from renderer.css, renderer/*.js, and d3
#
# Usage: cmake -DCSS_FILE=... -DJS_FILES=\"a.js;b.js\" -DD3_FILE=... -DOUT_FILE=... -P GenAssets.cmake

file(READ "${CSS_FILE}" CSS_CONTENT)
set(JS_CONTENT "")
string(REPLACE ",,FCC_VIZ_SEP,," ";" JS_FILE_LIST "${JS_FILES}")
foreach(JS_FILE IN LISTS JS_FILE_LIST)
  file(READ "${JS_FILE}" CUR_JS_CONTENT)
  string(APPEND JS_CONTENT "${CUR_JS_CONTENT}\n")
endforeach()
file(READ "${D3_FILE}" D3_CONTENT)

# Escape backslashes and closing paren sequences that would break R"delim(...)delim"
# We use VIZ_DELIM as delimiter which is unlikely to appear in CSS/JS
file(WRITE "${OUT_FILE}"
"// Auto-generated from renderer.css, renderer.js, and d3.min.js. Do not edit.\n"
"#ifndef FCC_VIZ_VIZASSETS_H\n"
"#define FCC_VIZ_VIZASSETS_H\n"
"\n"
"namespace fcc {\n"
"namespace viz {\n"
"\n"
"static const char *RENDERER_CSS = R\"VIZ_DELIM(\n"
"${CSS_CONTENT}"
"\n)VIZ_DELIM\";\n"
"\n"
"static const char *RENDERER_JS = R\"VIZ_DELIM(\n"
"${JS_CONTENT}"
"\n)VIZ_DELIM\";\n"
"\n"
"static const char *D3_MIN_JS = R\"VIZ_DELIM(\n"
"${D3_CONTENT}"
"\n)VIZ_DELIM\";\n"
"\n"
"} // namespace viz\n"
"} // namespace fcc\n"
"\n"
"#endif // FCC_VIZ_VIZASSETS_H\n"
)
