cmake_minimum_required(VERSION 3.20)
project(edge_update LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(edge_update_func main_func.cpp)
add_executable(edge_update_inline main_inline.cpp)
