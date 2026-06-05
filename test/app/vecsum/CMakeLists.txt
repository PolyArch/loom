cmake_minimum_required(VERSION 3.20)
project(vecsum LANGUAGES C)

add_executable(vecsum_func main_func.c)
add_executable(vecsum_inline main_inline.c)
