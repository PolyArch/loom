cmake_minimum_required(VERSION 3.20)
project(fft_butterfly LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(fft_butterfly_func main_func.cpp)
add_executable(fft_butterfly_inline main_inline.cpp)
