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

OPTION(PLUGINS_BUILD_FIBERSTOVOXELS "Build Fibers to Voxels Plugin" ON)
IF(PLUGINS_BUILD_FIBERSTOVOXELS)

CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )
PROJECT( FibersToVoxelsPlugin )

SET( BMIA_FIBERSTOVOXELS_PLUGIN )

FIND_PACKAGE( VTK REQUIRED )
INCLUDE( ${VTK_USE_FILE} )

FIND_PACKAGE( Qt4 REQUIRED )
INCLUDE( ${QT_USE_FILE} )

INCLUDE_DIRECTORIES(
	${CMAKE_CURRENT_BINARY_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}
#	${CMAKE_CURRENT_SOURCE_DIR}/../../../stable/DTIReader
	)

#LINK_DIRECTORIES(
#	${CMAKE_CURRENT_BINARY_DIR}/../../../../stable/DTIReader/Debug
#	)

SET( SRCS
	FibersToVoxelsPlugin.cxx
	FibersToVoxelsPlugin.h
	vtkMathExtensions.cxx
	vtkMathExtensions.h
	vtkStreamlineToVoxelDensity.cxx
	vtkStreamlineToVoxelDensity.h
#    vtkFiberSortFilter.cxx
#    vtkFiberSortFilter.h
	)

QT4_WRAP_CPP( MOC_SRCS
	FibersToVoxelsPlugin.h
	)

ADD_LIBRARY( FibersToVoxelsPlugin SHARED
	${SRCS} ${MOC_SRCS}
	)
    
TARGET_LINK_LIBRARIES( FibersToVoxelsPlugin
#	DTIReaderPlugin
	bmia_plugin
    bmia_libs
	)

ENDIF(PLUGINS_BUILD_FIBERSTOVOXELS)
