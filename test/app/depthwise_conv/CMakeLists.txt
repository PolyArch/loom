cmake_minimum_required(VERSION 3.20)
project(depthwise_conv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(depthwise_conv_func main_func.cpp)
add_executable(depthwise_conv_inline main_inline.cpp)
