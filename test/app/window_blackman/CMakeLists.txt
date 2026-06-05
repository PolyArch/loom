cmake_minimum_required(VERSION 3.20)
project(window_blackman LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(window_blackman_func main_func.cpp)
add_executable(window_blackman_inline main_inline.cpp)
