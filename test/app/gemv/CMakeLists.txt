cmake_minimum_required(VERSION 3.20)
project(gemv LANGUAGES C)

add_executable(gemv_func main_func.c)
add_executable(gemv_inline main_inline.c)
