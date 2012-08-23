OPTION(PLUGINS_BUILD_AXESVISUALIZATION "Build Axes Visualization Plugin" ON)

if (PLUGINS_BUILD_AXESVISUALIZATION)

set( BMIA_AXESVISUALIZATION_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
    AxesVisualizationPlugin.cxx
    AxesVisualizationPlugin.h
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${BMIA_LIBS_SOURCE_DIR})

qt4_wrap_ui(UiHeaders AxesVisualization.ui)
qt4_wrap_cpp( MOC_SRCS AxesVisualizationPlugin.h )
set_source_files_properties(AxesVisualizationPlugin.cxx PROPERTIES OBJECT_DEPENDS "${UiHeaders}")

link_directories(${BMIA_LIBS_BINARY_DIR})
add_library( AxesVisualizationPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( AxesVisualizationPlugin bmia_plugin bmia_libs ) 

endif(PLUGINS_BUILD_AXESVISUALIZATION)