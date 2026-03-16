# GenAssets.cmake - Generate VizAssets.h from renderer.css and renderer.js
#
# Usage: cmake -DCSS_FILE=... -DJS_FILE=... -DOUT_FILE=... -P GenAssets.cmake

file(READ "${CSS_FILE}" CSS_CONTENT)
file(READ "${JS_FILE}" JS_CONTENT)

# Escape backslashes and closing paren sequences that would break R"delim(...)delim"
# We use VIZ_DELIM as delimiter which is unlikely to appear in CSS/JS
file(WRITE "${OUT_FILE}"
"// Auto-generated from renderer.css and renderer.js. Do not edit.\n"
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
"} // namespace viz\n"
"} // namespace fcc\n"
"\n"
"#endif // FCC_VIZ_VIZASSETS_H\n"
)
