cmake_minimum_required(VERSION 3.20)
project(prefix_sum_inclusive LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(prefix_sum_inclusive_func main_func.cpp)
add_executable(prefix_sum_inclusive_inline main_inline.cpp)
