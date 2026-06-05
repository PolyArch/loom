cmake_minimum_required(VERSION 3.20)
project(loom_app_delta_encode C)

add_executable(delta_encode_func main_func.c)
add_executable(delta_encode_inline main_inline.c)
