cmake_minimum_required(VERSION 3.20)
project(normalize_vec3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(normalize_vec3_func main_func.cpp)
add_executable(normalize_vec3_inline main_inline.cpp)
