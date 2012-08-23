OPTION(PLUGINS_BUILD_POLYDATAVISUALIZATION "Build PolyData Visualization Plugin" ON)

if(PLUGINS_BUILD_POLYDATAVISUALIZATION)

set( BMIA_POLYDATAVIS_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
    PolyDataVisualizationPlugin.cxx
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

qt4_wrap_ui(UiHeaders polydata.ui)
qt4_wrap_cpp( MOC_SRCS PolyDataVisualizationPlugin.h )
set_source_files_properties(PolyDataVisualizationPlugin.cxx PROPERTIES OBJECT_DEPENDS "${UiHeaders}")


add_library( PolyDataVisualizationPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( PolyDataVisualizationPlugin bmia_libs )

endif(PLUGINS_BUILD_POLYDATAVISUALIZATION)
