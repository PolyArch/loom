#!/usr/bin/env python3
"""Identity-critical OR-Tools build configuration owned by Loom."""

OR_TOOLS_SEMANTIC_CMAKE_ARGS = (
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_INSTALL_LIBDIR=lib",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DBUILD_CXX=ON",
    "-DBUILD_PYTHON=OFF",
    "-DBUILD_JAVA=OFF",
    "-DBUILD_DOTNET=OFF",
    "-DBUILD_FLATZINC=OFF",
    "-DBUILD_MATH_OPT=OFF",
    "-DBUILD_SAMPLES=OFF",
    "-DBUILD_EXAMPLES=OFF",
    "-DBUILD_DOC=OFF",
    "-DBUILD_TESTING=OFF",
    "-DBUILD_DEPS=ON",
    "-DINSTALL_BUILD_DEPS=ON",
    "-DUSE_COINOR=OFF",
    "-DUSE_GLPK=OFF",
    "-DUSE_HIGHS=OFF",
    "-DUSE_PDLP=OFF",
    "-DUSE_SCIP=OFF",
)
