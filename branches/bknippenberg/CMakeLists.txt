cmake_minimum_required(VERSION 2.6)

OPTION(BMIA_BUILD_TESTS "Build test programs. Leave this ON for now" ON)
OPTION(BMIA_BUILD_FIBERTRACKING_DEFAULT "Build default (CPU) Fiber Tracking plugin" ON)
OPTION(BMIA_BUILD_FIBERTRACKING_CUDA "Build CUDA-Enabled Fiber Tracking plugin (requires CUDA)" OFF)

#OPTION(BUILD_ILLUSTRATIVE_CLUSTERS_PLUGIN "Build Illustrative Clusters Plugin (Requires GLEW)" ON)

SET( CMAKE_BUILD_TYPE "Debug Release RelWithDebInfo MinSizeRel" CACHE STRING
    "Choose the type of build" FORCE )

add_subdirectory( tool )
add_subdirectory( libs )
add_subdirectory( plugins )
