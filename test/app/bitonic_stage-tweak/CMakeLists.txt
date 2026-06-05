cmake_minimum_required(VERSION 3.20)
project(bitonic_stage_tweak LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(bitonic_stage-tweak_func main_func.cpp)
add_executable(bitonic_stage-tweak_inline main_inline.cpp)
