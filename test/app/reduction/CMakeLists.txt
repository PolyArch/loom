cmake_minimum_required(VERSION 3.20)
project(reduction LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_executable(reduction_func main_func.c)
add_executable(reduction_inline main_inline.c)
