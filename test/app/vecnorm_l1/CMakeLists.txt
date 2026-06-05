cmake_minimum_required(VERSION 3.20)
project(loom_app_vecnorm_l1 C)

add_executable(vecnorm_l1_func main_func.c)
add_executable(vecnorm_l1_inline main_inline.c)
