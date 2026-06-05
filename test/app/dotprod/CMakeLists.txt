cmake_minimum_required(VERSION 3.20)
project(dotprod LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(dotprod_func main_func.cpp)
add_executable(dotprod_inline main_inline.cpp)
