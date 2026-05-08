cmake_minimum_required(VERSION 3.20)
project(prefix_sum LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_executable(prefix_sum_func main_func.c)
add_executable(prefix_sum_inline main_inline.c)
