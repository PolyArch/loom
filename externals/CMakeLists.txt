# Suppress warnings from external dependencies
add_compile_options(-w)

# Build LLVM/MLIR/CIRCT (CIRCT integrated as LLVM external project)
add_subdirectory(llvm/llvm)
