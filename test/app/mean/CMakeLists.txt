cmake_minimum_required(VERSION 3.20)
project(mean LANGUAGES C)

add_executable(mean_func main_func.c)
add_executable(mean_inline main_inline.c)
