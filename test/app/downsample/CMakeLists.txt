cmake_minimum_required(VERSION 3.20)
project(loom_app_downsample C)

add_executable(downsample_func main_func.c)
add_executable(downsample_inline main_inline.c)
