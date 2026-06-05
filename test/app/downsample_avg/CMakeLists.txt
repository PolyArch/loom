cmake_minimum_required(VERSION 3.20)
project(loom_app_downsample_avg C)

add_executable(downsample_avg_func main_func.c)
add_executable(downsample_avg_inline main_inline.c)
