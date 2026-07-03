cmake_minimum_required(VERSION 3.20)
project(wildcard_match LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(wildcard_match_func main_func.cpp)
add_executable(wildcard_match_inline main_inline.cpp)
