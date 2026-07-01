cmake_minimum_required(VERSION 3.20)
project(database_join LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(database_join_func main_func.cpp)
add_executable(database_join_inline main_inline.cpp)
