cmake_minimum_required(VERSION 3.20)
project(hist_bin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(hist_bin_func main_func.cpp)
add_executable(hist_bin_inline main_inline.cpp)
