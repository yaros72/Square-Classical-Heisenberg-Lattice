cmake_minimum_required(VERSION 3.5)
project(Square_Skyrmion_Annealing)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES main.cpp)

#set(TRIQS_PATH ~/TRIQS)
list(APPEND CMAKE_MODULE_PATH ${TRIQS_PATH}/share/triqs/cmake)
find_package(TRIQS REQUIRED)
link_libraries(${TRIQS_LIBRARY_ALL})
include_directories(${TRIQS_INCLUDE_ALL})

add_executable(Square_Skyrmion_Annealing ${SOURCE_FILES})
