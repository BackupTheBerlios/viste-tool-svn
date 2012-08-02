set( BMIA_TRANSFORMATION_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
	TransformationPlugin.cxx
	TransformationPlugin.h
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${BMIA_LIBS_SOURCE_DIR})

qt4_wrap_ui(UiHeaders Transformation.ui)
qt4_wrap_cpp(MOC_SRCS TransformationPlugin.h)
set_source_files_properties(TransformationPlugin.cxx PROPERTIES OBJECT_DEPENDS "${UiHeaders}")

link_directories(${BMIA_LIBS_BINARY_DIR})
add_library( TransformationPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( TransformationPlugin bmia_plugin bmia_data bmia_libs vtkTensorMath )
