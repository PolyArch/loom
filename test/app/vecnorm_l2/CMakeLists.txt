cmake_minimum_required(VERSION 3.20)
project(loom_app_vecnorm_l2 C)

add_executable(vecnorm_l2_func main_func.c)
add_executable(vecnorm_l2_inline main_inline.c)
