cmake_minimum_required(VERSION 3.20)
project(jacobi_stencil_7pt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(jacobi_stencil_7pt_func main_func.cpp)
add_executable(jacobi_stencil_7pt_inline main_inline.cpp)
