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

PROJECT( BMIA_HWSHADING )

find_package(VTK)
if(VTK_FOUND)
  include(${VTK_USE_FILE})
endif(VTK_FOUND)


include_directories(${BMIA_LIBS_SOURCE_DIR})

SET( HWShading_classes
vtkShaderBase
vtkShaderBaseHandle
vtkShaderObject
vtkVertexShader
vtkFragmentShader
vtkShaderObjectCollection
vtkShaderUniform
vtkUniformFloat
vtkUniformVec2
vtkUniformVec3
vtkUniformIvec3
vtkUniformVec4
vtkUniformBool
vtkUniformInt
vtkUniformSampler
vtkShaderUniformCollection
vtkMyShaderProgram
vtkMyShaderProgramReader
vtkAnisotropicLightingSP
vtkFBO
vtkShadowMappingSP
vtkShadowMappingHelper
vtkShadowRenderer
vtkAnisoLiShadowMapSP
vtkShadowMappingHelperLines
vtkFiberMapper
#vtkEigensystemToGPU
)

FOREACH(class ${HWShading_classes})
  SET( HWShading_SRCS ${HWShading_SRCS} ${class}.h ${class}.cxx )
ENDFOREACH(class)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/Shaders)
INCLUDE(ShaderText.cmake)
SHADER_TO_HEADER(	"AnisotropicLightingVertexText"
			"Shaders/AnisotropicLighting.vert"
			 "${CMAKE_CURRENT_BINARY_DIR}/Shaders/AnisotropicLightingVertexText.h"
		)

SHADER_TO_HEADER(	"AnisotropicLightingFragmentText"
			"Shaders/AnisotropicLighting.frag"
			 "${CMAKE_CURRENT_BINARY_DIR}/Shaders/AnisotropicLightingFragmentText.h"
		)
SHADER_TO_HEADER(	"ShadowMappingVertexText"
			"Shaders/ShadowMapping.vert"
			"${CMAKE_CURRENT_BINARY_DIR}/Shaders/ShadowMappingVertexText.h"
		)
SHADER_TO_HEADER(	"ShadowMappingFragmentText"
			"Shaders/ShadowMapping.frag"
			"${CMAKE_CURRENT_BINARY_DIR}/Shaders/ShadowMappingFragmentText.h"
		)
SHADER_TO_HEADER(	"SpotlightFunctionsText"
			"Shaders/Spotlight.func"
			"${CMAKE_CURRENT_BINARY_DIR}/Shaders/SpotlightFunctionsText.h"
		)
SHADER_TO_HEADER(	"AnisotropicLightingFunctionsText"
			"Shaders/AnisotropicLighting.func"
			"${CMAKE_CURRENT_BINARY_DIR}/Shaders/AnisotropicLightingFunctionsText.h"
		)
SHADER_TO_HEADER(	"AnisotropicLightingShadowMappingFragmentText"
			"Shaders/AnisoLiShadowMap.frag"
			"${CMAKE_CURRENT_BINARY_DIR}/Shaders/AnisotropicLightingShadowMappingFragmentText.h"
		)
SHADER_TO_HEADER(	"AnisotropicLightingShadowMappingVertexText"
			"Shaders/AnisoLiShadowMap.vert"
			"${CMAKE_CURRENT_BINARY_DIR}/Shaders/AnisotropicLightingShadowMappingVertexText.h"
		)
SHADER_TO_HEADER(	"BuildShadowMapVertexText"
			"Shaders/BuildShadowMap.vert"
			"${CMAKE_CURRENT_BINARY_DIR}/Shaders/BuildShadowMapVertexText.h"
		)
SHADER_TO_HEADER(	"BuildShadowMapFragmentText"
			"Shaders/BuildShadowMap.frag"
			"${CMAKE_CURRENT_BINARY_DIR}/Shaders/BuildShadowMapFragmentText.h"
		)
SHADER_TO_HEADER(	"BuildShadowMapLinesVertexText"
			"Shaders/BuildShadowMapLines.vert"
			"${CMAKE_CURRENT_BINARY_DIR}/Shaders/BuildShadowMapLinesVertexText.h"
		)
SHADER_TO_HEADER(	"BuildShadowMapLinesFragmentText"
			"Shaders/BuildShadowMapLines.frag"
			"${CMAKE_CURRENT_BINARY_DIR}/Shaders/BuildShadowMapLinesFragmentText.h"
		)

ADD_LIBRARY( vtkHWShading ${HWShading_SRCS} )
#TARGET_LINK_LIBRARIES( vtkHWShading vtkRendering ${GLEWLIB} vtkHelpers )
TARGET_LINK_LIBRARIES( vtkHWShading vtkHelpers vtkHARDI vtkRendering )

#SUBDIRS(Shaders)
