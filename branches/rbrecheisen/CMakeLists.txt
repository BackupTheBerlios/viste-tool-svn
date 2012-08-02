cmake_minimum_required(VERSION 2.6)

find_package(VTK)
if(VTK_FOUND)
  include(${VTK_USE_FILE})
endif(VTK_FOUND)

find_package(Qt4 REQUIRED)
SET(QT_USE_QTXML true)
include(${QT_USE_FILE})

OPTION(TESTING_BUILD_TESTS "Build test programs." OFF)

SET(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build" FORCE)

IF(UNIX)
SET(CMAKE_CXX_FLAGS  "-fPIC" CACHE STRING "Add CXX compiler flags" FORCE)
SET(CMAKE_C_FLAGS    "-fPIC" CACHE STRING "Add C compiler flags" FORCE)
ENDIF(UNIX)

add_subdirectory( tool )
add_subdirectory( libs )
add_subdirectory( plugins )
