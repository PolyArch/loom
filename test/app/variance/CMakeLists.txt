cmake_minimum_required(VERSION 3.20)
project(loom_app_variance C)

add_executable(variance_func main_func.c)
add_executable(variance_inline main_inline.c)
