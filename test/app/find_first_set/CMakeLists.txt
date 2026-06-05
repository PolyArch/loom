cmake_minimum_required(VERSION 3.20)
project(find_first_set LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(find_first_set_func main_func.cpp)
add_executable(find_first_set_inline main_inline.cpp)
