cmake_minimum_required(VERSION 3.20)
project(fir_filter_stateful LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(fir_filter_stateful_func main_func.cpp)
add_executable(fir_filter_stateful_inline main_inline.cpp)
