cmake_minimum_required(VERSION 2.6)

OPTION(TESTING_BUILD_TESTS "Build test programs." OFF)

SET(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build" FORCE)
SET(CMAKE_CXX_FLAGS  "-fPIC" CACHE STRING "Add CXX compiler flags" FORCE)
SET(CMAKE_C_FLAGS    "-fPIC" CACHE STRING "Add C compiler flags" FORCE)

add_subdirectory( tool )
add_subdirectory( libs )
add_subdirectory( plugins )
