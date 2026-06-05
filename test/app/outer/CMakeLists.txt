cmake_minimum_required(VERSION 3.20)
project(outer LANGUAGES C)

add_executable(outer_func main_func.c)
add_executable(outer_inline main_inline.c)
