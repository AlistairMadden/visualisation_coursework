
cmake_minimum_required(VERSION 2.8)

project (visualisation_coursework)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

set(CMAKE_CXX_FLAGS "-std=c++0x -pthread")

add_executable (visualisation_coursework visualisation_coursework.cpp)

if(VTK_LIBRARIES)
  target_link_libraries(visualisation_coursework ${VTK_LIBRARIES})
  MESSAGE(STATUS "VTK_LIBRARIES: " ${VTK_LIBRARIES})
else()
  target_link_libraries(visualisation_coursework vtkHybrid vtkWidgets)
endif()
