cmake_minimum_required(VERSION 3.20)
project(loom_app_integrate_trapz C)

add_executable(integrate_trapz_func main_func.c)
add_executable(integrate_trapz_inline main_inline.c)
