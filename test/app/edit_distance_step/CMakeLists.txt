cmake_minimum_required(VERSION 3.20)
project(edit_distance_step LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(edit_distance_step_func main_func.cpp)
add_executable(edit_distance_step_inline main_inline.cpp)
