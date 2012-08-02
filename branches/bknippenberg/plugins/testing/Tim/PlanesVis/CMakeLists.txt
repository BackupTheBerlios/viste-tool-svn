set( BMIA_PLANESVIS_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
	vtkMEVColoringFilter.cxx
	vtkMEVColoringFilter.h
	PlanesVisPlugin.cxx
	PlanesVisPlugin.h
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${BMIA_LIBS_SOURCE_DIR})

qt4_wrap_ui(UiHeaders PlanesVisualization.ui)
qt4_wrap_cpp( MOC_SRCS PlanesVisPlugin.h )
set_source_files_properties(PlanesVisPlugin.cxx PROPERTIES OBJECT_DEPENDS "${UiHeaders}")

link_directories(${BMIA_LIBS_BINARY_DIR})
add_library( PlanesVisPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( PlanesVisPlugin bmia_plugin bmia_libs )
