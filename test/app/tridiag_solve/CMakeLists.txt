cmake_minimum_required(VERSION 3.20)
project(tridiag_solve LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(tridiag_solve_func main_func.cpp)
add_executable(tridiag_solve_inline main_inline.cpp)
