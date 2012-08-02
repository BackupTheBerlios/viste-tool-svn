set( BMIA_CONEVIS_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
    ConeVisualizationPlugin.cxx
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

qt4_wrap_ui(UiHeaders cone.ui)
qt4_wrap_cpp( MOC_SRCS ConeVisualizationPlugin.h )
set_source_files_properties(ConeVisualizationPlugin.cxx PROPERTIES OBJECT_DEPENDS "${UiHeaders}")

add_library( ConeVisualizationPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( ConeVisualizationPlugin bmia_libs )
