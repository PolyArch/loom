cmake_minimum_required(VERSION 3.20)
project(matvec LANGUAGES C)

add_executable(matvec_func main_func.c)
add_executable(matvec_inline main_inline.c)
