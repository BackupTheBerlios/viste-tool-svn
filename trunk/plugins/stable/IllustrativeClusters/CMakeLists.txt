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

OPTION(PLUGINS_BUILD_ILLUSTRATIVECLUSTERS "Build Illustrative Clusters Plugin (Requires GLEW)" OFF)

if (PLUGINS_BUILD_ILLUSTRATIVECLUSTERS)

set( BMIA_ILLUSTRATIVECLUSTERS_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

find_package(OpenGL REQUIRED)

IF( WIN32 )
  SET( GLEW_lib glew32.lib )
ELSE( LINUX )
  SET( GLEW_lib libGLEW.so )
ENDIF()

SET( OpenGLLibs
  ${OPENGL_LIBRARIES}
  ${GLEW_lib}
)

INCLUDE_DIRECTORIES(
  ${OPENGL_INCLUDE_DIR}
)

set(SRCS
	IllustrativeClustersPlugin.cxx
	IllustrativeClustersPlugin.h
	IllustrativeCluster.cxx
	IllustrativeCluster.h
	vtkIllustrativeFiberBundleMapper.cxx
	vtkIllustrativeFiberBundleMapper.h
	IllustrativeClusterDisplacement.cxx
	IllustrativeClusterDisplacement.h
	
	Math/Vector3.cxx
	Math/Vector3.h
	Math/Matrix3.cxx
	Math/Matrix3.h
	Math/ConvexHull.cxx
	Math/ConvexHull.h
	Math/OrientedBoundingBox.cxx
	Math/OrientedBoundingBox.h
	Math/SeparatingAxis.cxx
	Math/SeparatingAxis.h
	
	GpuBuffers/VertexBuffer.cxx
	GpuBuffers/VertexBuffer.h
	GpuBuffers/VertexBufferDeclaration.cxx
	GpuBuffers/VertexBufferDeclaration.h
	GpuBuffers/VertexBufferBinding.cxx
	GpuBuffers/VertexBufferBinding.h
	GpuBuffers/FrameBuffer.cxx
	GpuBuffers/FrameBuffer.h
	GpuBuffers/FrameBufferDeclaration.cxx
	GpuBuffers/FrameBufferDeclaration.h
	GpuBuffers/FrameBufferBinding.cxx
	GpuBuffers/FrameBufferBinding.h
	GpuBuffers/TransformFeedback.cxx
	GpuBuffers/TransformFeedback.h
	GpuBuffers/TransformFeedbackQuery.cxx
	GpuBuffers/TransformFeedbackQuery.h
	
	Enumerators/MapEnumerator.cxx
	Enumerators/MapEnumerator.h
	Enumerators/VectorEnumerator.cxx
	Enumerators/VectorEnumerator.h

	GpuPrograms/GpuProgram.cxx
	GpuPrograms/GpuProgram.h
	GpuPrograms/GpuShader.cxx
	GpuPrograms/GpuShader.h
	
	GpuTextures/Texture2D.cxx
	GpuTextures/Texture2D.h
	GpuTextures/Texture3D.cxx
	GpuTextures/Texture3D.h
	GpuTextures/TextureDeclaration.cxx
	GpuTextures/TextureDeclaration.h

)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${BMIA_LIBS_SOURCE_DIR})

qt4_wrap_ui(UiHeaders IllustrativeClusters.ui)
qt4_wrap_cpp(MOC_SRCS IllustrativeClustersPlugin.h)
set_source_files_properties(IllustrativeClustersPlugin.cxx PROPERTIES OBJECT_DEPENDS "${UiHeaders}")

link_directories(${BMIA_LIBS_BINARY_DIR})

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/Shaders)
INCLUDE(ShaderText.cmake)
SHADER_TO_HEADER(	"BuildHaloFins_GeometryShaderCode"
			"Shaders/BuildHaloFins.geom"
			 "${CMAKE_CURRENT_BINARY_DIR}/Shaders/BuildHaloFins.geom.h"
		)

SHADER_TO_HEADER(	"BuildHaloLines_GeometryShaderCode"
			"Shaders/BuildHaloLines.geom"
			 "${CMAKE_CURRENT_BINARY_DIR}/Shaders/BuildHaloLines.geom.h"
		)
SHADER_TO_HEADER(	"DilateSilhouetteContours_FragmentShaderCode"
			"Shaders/DilateSilhouetteContours.frag"
			"${CMAKE_CURRENT_BINARY_DIR}/Shaders/DilateSilhouetteContours.frag.h"
		)
SHADER_TO_HEADER(	"DilateSilhouetteContours_VertexShaderCode"
			"Shaders/DilateSilhouetteContours.vert"
			"${CMAKE_CURRENT_BINARY_DIR}/Shaders/DilateSilhouetteContours.vert.h"
		)
SHADER_TO_HEADER(	"InkHaloFins_FragmentShaderCode"
			"Shaders/InkHaloFins.frag"
			"${CMAKE_CURRENT_BINARY_DIR}/Shaders/InkHaloFins.frag.h"
		)
SHADER_TO_HEADER(	"InkHaloLines_FragmentShaderCode"
			"Shaders/InkHaloLines.frag"
			"${CMAKE_CURRENT_BINARY_DIR}/Shaders/InkHaloLines.frag.h"
		)
SHADER_TO_HEADER(	"Passthrough_VertexShaderCode"
			"Shaders/Passthrough.vert"
			"${CMAKE_CURRENT_BINARY_DIR}/Shaders/Passthrough.vert.h"
		)

add_library( IllustrativeClustersPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( IllustrativeClustersPlugin bmia_libs ${OpenGLLibs} vtkVolumeRendering vtkIO vtkWidgets vtkGraphics vtkRendering)

endif(PLUGINS_BUILD_ILLUSTRATIVECLUSTERS)