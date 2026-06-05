cmake_minimum_required(VERSION 3.20)
project(crc32 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(crc32_func main_func.cpp)
add_executable(crc32_inline main_inline.cpp)
