cmake_minimum_required(VERSION 2.6)

project(BMIA_TOOL)

find_package(VTK)
if(VTK_FOUND)
  include(${VTK_USE_FILE})
endif(VTK_FOUND)

find_package(Qt4 REQUIRED)
SET(QT_USE_QTXML true)
include(${QT_USE_FILE})

option(BMIA_BUILD_TESTS "Build test programs" OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BMIA_TOOL_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BMIA_TOOL_BINARY_DIR}/bin)

#include_directories(${CMAKE_CURRENT_SOURCE_DIR})
#include_directories(${CMAKE_CURRENT_BINARY_DIR}) # for ui_mainwindow.h
include_directories(${BMIA_TOOL_SOURCE_DIR})
#include_directories(${BMIA_TOOL_BINARY_DIR}/gui) # for ui_mainwindow.h
include_directories(${BMIA_TOOL_BINARY_DIR}) # for ui_mainwindow.h

add_subdirectory(data)
add_subdirectory(plugin)
add_subdirectory(gui)
add_subdirectory(core)

if (BMIA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(data/test)
  add_subdirectory(core/test)
#  add_subdirectory(plugin/test)
endif (BMIA_BUILD_TESTS)

#add_library(bmia_tool)	# an empty library, linking to the "sub" libraries of the tool
#target_link_libraries(bmia_tool bmia_data bmia_core bmia_plugin)
