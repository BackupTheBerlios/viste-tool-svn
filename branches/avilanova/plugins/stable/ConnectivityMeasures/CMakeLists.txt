OPTION(PLUGINS_BUILD_CONNECTIVITYMEASURES "Build Connectivity Measures Plugin" ON)

if (PLUGINS_BUILD_CONNECTIVITYMEASURES)

set( BMIA_CONNECTIVITYMEASURES_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
	ConnectivityMeasuresPlugin.cxx
	ConnectivityMeasuresPlugin.h
	vtkGenericConnectivityMeasureFilter.cxx
	vtkGenericConnectivityMeasureFilter.h
	vtkGeodesicConnectionStrengthFilter.cxx
	vtkGeodesicConnectionStrengthFilter.h
	vtkFiberRankingFilter.cxx
	vtkFiberRankingFilter.h
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${BMIA_LIBS_SOURCE_DIR})

qt4_wrap_ui(UiHeaders ConnectivityMeasures.ui)
qt4_wrap_cpp(MOC_SRCS ConnectivityMeasuresPlugin.h)
set_source_files_properties(ConnectivityMeasuresPlugin.cxx PROPERTIES OBJECT_DEPENDS "${UiHeaders}")

link_directories(${BMIA_LIBS_BINARY_DIR})
add_library( ConnectivityMeasuresPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( ConnectivityMeasuresPlugin bmia_libs vtkTensorMath )

endif(PLUGINS_BUILD_CONNECTIVITYMEASURES)