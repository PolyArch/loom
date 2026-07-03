cmake_minimum_required(VERSION 3.13)
project(gauss_seidel_step_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(gauss_seidel_step_func main_func.cpp)
add_executable(gauss_seidel_step_inline main_inline.cpp)
