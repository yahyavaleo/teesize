cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(teesize)

set(CMAKE_PREFIX_PATH "/home/yahya/Documents/Projects/teesize/prod/libtorch")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(teesize teesize.cpp)
target_link_libraries(teesize "${TORCH_LIBRARIES}" ${OpenCV_LIBS})

set_property(TARGET teesize PROPERTY CXX_STANDARD 17)
