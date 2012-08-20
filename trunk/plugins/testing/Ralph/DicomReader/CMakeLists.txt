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

CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )
SET( PROJECT_NAME DicomReaderPlugin )
SET( BMIA_DICOMREADER_PLUGIN )
PROJECT( ${PROJECT_NAME} )

#----------------------------------------------------------
# Find QT package
#----------------------------------------------------------
FIND_PACKAGE( Qt4 REQUIRED )
INCLUDE( ${QT_USE_FILE} )

#----------------------------------------------------------
# Find GDCM package
#----------------------------------------------------------
FIND_PACKAGE( GDCM REQUIRED )
INCLUDE( ${GDCM_USE_FILE} )

#----------------------------------------------------------
# Include and link directories
#----------------------------------------------------------
INCLUDE_DIRECTORIES(
	${CMAKE_CURRENT_BINARY_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}
	)

#----------------------------------------------------------
# Run MOC compiler
#----------------------------------------------------------
QT4_WRAP_CPP( SOURCES_MOC
	DicomReaderPlugin.h
	)

#----------------------------------------------------------
# Add sources and headers to library
#----------------------------------------------------------
ADD_LIBRARY( ${PROJECT_NAME} SHARED ${SOURCES_MOC}
	DicomReaderPlugin.cxx
	DicomReaderPlugin.h
	)
	
#----------------------------------------------------------
# Specify link libraries
#----------------------------------------------------------
TARGET_LINK_LIBRARIES( ${PROJECT_NAME}
	gdcmCommon gdcmDICT gdcmIOD gdcmDSED gdcmjpeg12
	gdcmjpeg16 gdcmjpeg8 gdcmMSFF gdcmopenjpeg gdcmzlib gdcmexpat 
	vtkgdcm vtkVolumeRendering vtkIO vtkWidgets vtkGraphics vtkRendering
	bmia_plugin 
	)
