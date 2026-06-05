cmake_minimum_required(VERSION 3.20)
project(moving_avg LANGUAGES C)

add_executable(moving_avg_func main_func.c)
add_executable(moving_avg_inline main_inline.c)
