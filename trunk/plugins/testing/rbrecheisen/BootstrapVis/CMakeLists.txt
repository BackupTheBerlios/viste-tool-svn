set( BMIA_BOOTSTRAPVIS_PLUGIN )
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
	BootstrapVisPlugin.cxx
	vtkColor4.cxx
	vtkFiberConfidenceMapper.cxx
	vtkFiberTubeMapper.cxx
	vtkDistanceMeasure.cxx
	vtkDistanceMeasureClosestPointDistance.cxx
	vtkDistanceMeasureEndPointDistance.cxx
	vtkDistanceMeasureMeanOfClosestPointDistances.cxx
	vtkDistanceTable.cxx
	Math/Vector3.cxx
	Math/Matrix3.cxx
	Math/ConvexHull.cxx
	Math/OrientedBoundingBox.cxx
	Math/SeparatingAxis.cxx
	Enumerators/MapEnumerator.cxx
	Enumerators/VectorEnumerator.cxx
	GpuPrograms/GpuProgram.cxx
	GpuPrograms/GpuShader.cxx
	GpuTextures/Texture2D.cxx
	GpuTextures/Texture3D.cxx
	GpuTextures/TextureDeclaration.cxx
	GpuBuffers/VertexBuffer.cxx
	GpuBuffers/VertexBufferDeclaration.cxx
	GpuBuffers/VertexBufferBinding.cxx
	GpuBuffers/FrameBuffer.cxx
	GpuBuffers/FrameBufferDeclaration.cxx
	GpuBuffers/FrameBufferBinding.cxx
	GpuBuffers/TransformFeedback.cxx
	GpuBuffers/TransformFeedbackQuery.cxx
)

INCLUDE_DIRECTORIES(
	${CMAKE_CURRENT_BINARY_DIR}
	${CMAKE_CURRENT_BINARY_DIR}/Shaders
	/usr/include )

LINK_DIRECTORIES(
	/usr/local/lib
	/usr/lib )

INCLUDE(ShaderText.cmake)

SHADER_TO_HEADER("BuildHaloFins_GeometryShaderCode"
	"Shaders/BuildHaloFins.geom"
	"${CMAKE_CURRENT_BINARY_DIR}/Shaders/BuildHaloFins.geom.h"
	)

SHADER_TO_HEADER("BuildHaloLines_GeometryShaderCode"
	"Shaders/BuildHaloLines.geom"
	"${CMAKE_CURRENT_BINARY_DIR}/Shaders/BuildHaloLines.geom.h"
	)

SHADER_TO_HEADER("InkHaloFins_FragmentShaderCode"
	"Shaders/InkHaloFins.frag"
	"${CMAKE_CURRENT_BINARY_DIR}/Shaders/InkHaloFins.frag.h"
	)
	
SHADER_TO_HEADER("InkHaloLines_FragmentShaderCode"
	"Shaders/InkHaloLines.frag"
	"${CMAKE_CURRENT_BINARY_DIR}/Shaders/InkHaloLines.frag.h"
	)
	
SHADER_TO_HEADER("Passthrough_VertexShaderCode"
	"Shaders/Passthrough.vert"
	"${CMAKE_CURRENT_BINARY_DIR}/Shaders/Passthrough.vert.h"
	)

SHADER_TO_HEADER("Default_VertexShaderCode"
	"Shaders/Default.vert"
	"${CMAKE_CURRENT_BINARY_DIR}/Shaders/Default.vert.h"
	)
	
SHADER_TO_HEADER("Lighting_FragmentShaderCode"
	"Shaders/Lighting.frag"
	"${CMAKE_CURRENT_BINARY_DIR}/Shaders/Lighting.frag.h"
	)
	
SHADER_TO_HEADER("DensityMap_VertexShaderCode"
	"Shaders/DensityMap.vert"
	"${CMAKE_CURRENT_BINARY_DIR}/Shaders/DensityMap.vert.h"
	)

SHADER_TO_HEADER("DensityMap_FragmentShaderCode"
	"Shaders/DensityMap.frag"
	"${CMAKE_CURRENT_BINARY_DIR}/Shaders/DensityMap.frag.h"
	)

SHADER_TO_HEADER("SilhouetteMap_VertexShaderCode"
	"Shaders/SilhouetteMap.vert"
	"${CMAKE_CURRENT_BINARY_DIR}/Shaders/SilhouetteMap.vert.h"
	)

SHADER_TO_HEADER("SilhouetteMap_FragmentShaderCode"
	"Shaders/SilhouetteMap.frag"
	"${CMAKE_CURRENT_BINARY_DIR}/Shaders/SilhouetteMap.frag.h"
	)

SHADER_TO_HEADER("ToScreen_VertexShaderCode"
	"Shaders/ToScreen.vert"
	"${CMAKE_CURRENT_BINARY_DIR}/Shaders/ToScreen.vert.h"
	)

SHADER_TO_HEADER("ToScreen_FragmentShaderCode"
	"Shaders/ToScreen.frag"
	"${CMAKE_CURRENT_BINARY_DIR}/Shaders/ToScreen.frag.h"
	)

SHADER_TO_HEADER("Erosion_VertexShaderCode"
	"Shaders/Erosion.vert"
	"${CMAKE_CURRENT_BINARY_DIR}/Shaders/Erosion.vert.h"
	)

SHADER_TO_HEADER("Erosion_FragmentShaderCode"
	"Shaders/Erosion.frag"
	"${CMAKE_CURRENT_BINARY_DIR}/Shaders/Erosion.frag.h"
	)

SHADER_TO_HEADER("DensitySmoothing_VertexShaderCode"
	"Shaders/DensitySmoothing.vert"
	"${CMAKE_CURRENT_BINARY_DIR}/Shaders/DensitySmoothing.vert.h"
	)

SHADER_TO_HEADER("DensitySmoothing_FragmentShaderCode"
	"Shaders/DensitySmoothing.frag"
	"${CMAKE_CURRENT_BINARY_DIR}/Shaders/DensitySmoothing.frag.h"
	)

#qt4_wrap_ui(UiHeaders bootstrap.ui)
qt4_wrap_cpp( MOC_SRCS BootstrapVisPlugin.h )
#set_source_files_properties(BootstrapVisPlugin.cxx PROPERTIES OBJECT_DEPENDS "${UiHeaders}")
add_library( BootstrapVisPlugin SHARED ${SRCS} ${MOC_SRCS} )
SET_TARGET_PROPERTIES( BootstrapVisPlugin PROPERTIES COMPILE_FLAGS -g )
target_link_libraries( BootstrapVisPlugin
	bmia_plugin
	GLEW )
