OPTION(PLUGINS_BUILD_FIBERVISUALIZATION "Build Fiber Visualization Plugin" ON)

if(PLUGINS_BUILD_FIBERVISUALIZATION)

set( BMIA_FIBERVIS_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

#subdirs(HWShading)

set(SRCS
    FiberVisualizationPlugin.cxx
    FiberVisualizationPlugin.h
	vtkStreamlineToStreamGeneralCylinder.cxx
	vtkStreamlineToStreamGeneralCylinder.h
	vtkStreamlineToStreamTube.cxx
	vtkStreamlineToStreamTube.h
	vtkStreamlineToHyperStreamline.cxx
	vtkStreamlineToHyperStreamline.h
	FiberVisualizationPipeline.cxx
	FiberVisualizationPipeline.h
	vtkStreamlineToHyperStreamPrisma.cxx
	vtkStreamlineToHyperStreamPrisma.h
	vtkMEVToStreamlineColor.cxx
	vtkMEVToStreamlineColor.h
	vtkDirectionToStreamlineColor.cxx
	vtkDirectionToStreamlineColor.h
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/HWShading)
include_directories(${BMIA_LIBS_SOURCE_DIR})

qt4_wrap_ui(UiHeaders FiberVisualization.ui)
qt4_wrap_cpp( MOC_SRCS FiberVisualizationPlugin.h )
set_source_files_properties(FiberVisualizationPlugin.cxx PROPERTIES OBJECT_DEPENDS "${UiHeaders}")

link_directories(${BMIA_LIBS_BINARY_DIR})
add_library( FiberVisualizationPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( FiberVisualizationPlugin bmia_libs ) #bmia_plugin bmia_data vtkHWShadingFibers vtkIO vtkGraphics )

endif(PLUGINS_BUILD_FIBERVISUALIZATION)