# Copyright (c) 2012, Biomedical Image Analysis Eindhoven (BMIA/e)
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without 
# modification, are permitted provided that the following conditions
# are met:
# 
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
# 
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in
#     the documentation and/or other materials provided with the 
#     distribution.
# 
#   - Neither the name of Eindhoven University of Technology nor the
#     names of its contributors may be used to endorse or promote 
#     products derived from this software without specific prior 
#     written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# 06.02.2012 Remove CMAKE_BUILD_TYPE assignment to Debug. 
# 06.02.2012 Add two messages to print the IDE selected and the status of 64bit compiler for windows. 
# 06.02.2012 CMAKE_BUILD_TYPE control and messages. 

cmake_minimum_required(VERSION 2.6)

# CMAKE_BUILD_TYPE default value is Release
IF(DEFINED CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of
build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug
Release.")
ELSE()
   SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build,
options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release.") 
ENDIF()

message( STATUS "CMAKE_GENERATOR: ${CMAKE_GENERATOR}")
message( STATUS "CMAKE_CL_64: ${CMAKE_CL_64}")
message( STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

OPTION(TESTING_BUILD_TESTS "Build test programs." OFF)

#SET(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build" FORCE)
#SET(CMAKE_OSX_ARCHITECTURES "x86_64;i386" CACHE STRING "Set OSX architectures" FORCE)
#SET(CMAKE_CXX_FLAGS "-fPIC" CACHE STRING "Add CXX compiler flags" FORCE)
SET(CMAKE_C_FLAGS "-fPIC" CACHE STRING "Add C compiler flags" FORCE)

add_subdirectory( tool )
add_subdirectory( libs )
add_subdirectory( plugins )

# CONFIGURE_FILE(../installation/package.bat  ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
# CONFIGURE_FILE(../installation/package.sh   ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
# CONFIGURE_FILE(../installation/package.nsi  ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
