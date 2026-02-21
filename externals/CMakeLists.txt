# Suppress all compiler warnings for external dependencies.
# Errors are still reported; only diagnostics at warning level are silenced.
add_compile_options(-w)

add_subdirectory(llvm-project/llvm)
