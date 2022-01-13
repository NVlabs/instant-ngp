cmake_minimum_required(VERSION 2.8)
project(path)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  # Enable C++11 mode on GCC / Clang
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()
add_executable(path_demo path_demo.cpp filesystem/path.h filesystem/resolver.h)
