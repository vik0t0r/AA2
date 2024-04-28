cmake_minimum_required(VERSION 3.23)
project(AA2 C)

set(CMAKE_C_STANDARD 99)

add_executable(AA2 main.c)
target_compile_options(AA2 PRIVATE -mavx)