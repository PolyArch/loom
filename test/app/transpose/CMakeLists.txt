cmake_minimum_required(VERSION 3.20)
project(loom_app_transpose C)

add_executable(transpose_func main_func.c)
add_executable(transpose_inline main_inline.c)
