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

project(BMIA_LIBS)

#include_directories(${BMIA_TOOL_SOURCE_DIR})
#include_directories(${BMIA_TOOL_BINARY_DIR})

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BMIA_TOOL_BINARY_DIR}/bin)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BMIA_TOOL_BINARY_DIR}/bin/plugins)
#set(BMIA_SHADER_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders)

add_subdirectory( Helpers )
add_subdirectory( HWShading )
add_subdirectory( HARDI )
add_subdirectory( TensorMath )
add_subdirectory( NIfTI )

# avoid CMake complaining that bmia_libs doesn't have any source files
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/dummy.c "")
add_library(bmia_libs "${CMAKE_CURRENT_BINARY_DIR}/dummy.c")

target_link_libraries(bmia_libs bmia_data bmia_plugin vtkIO vtkGraphics vtkRendering vtkVolumeRendering vtkWidgets vtkHelpers vtkHWShading)