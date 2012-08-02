OPTION(PLUGINS_BUILD_CLUSTERING "Build Clustering Plugin" ON)

if (PLUGINS_BUILD_CLUSTERING)

set( BMIA_CLUSTERING_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
	ClusteringPlugin.cxx
	ClusteringPlugin.h
	ClusteringSettingsIO.cxx
	ClusteringSettingsIO.h
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${BMIA_LIBS_SOURCE_DIR})

qt4_wrap_ui(UiHeaders Clustering.ui)
qt4_wrap_cpp(MOC_SRCS ClusteringPlugin.h)
set_source_files_properties(ClusteringPlugin.cxx PROPERTIES OBJECT_DEPENDS "${UiHeaders}")

link_directories(${BMIA_LIBS_BINARY_DIR})
add_library( ClusteringPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( ClusteringPlugin bmia_libs )

endif(PLUGINS_BUILD_CLUSTERING)