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
