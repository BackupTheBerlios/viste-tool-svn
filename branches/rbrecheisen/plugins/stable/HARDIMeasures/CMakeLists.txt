OPTION(PLUGINS_BUILD_HARDIMEASURES "Build HARDI Measures Plugin" ON)

if(PLUGINS_BUILD_HARDIMEASURES)

set( BMIA_HARDIMEASURES_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
	HARDIMeasuresPlugin.cxx
	HARDIMeasuresPlugin.h
	vtkDiscreteSphereToScalarVolumeFilter.cxx
	vtkDiscreteSphereToScalarVolumeFilter.h
)

qt4_wrap_cpp( MOC_SRCS HARDIMeasuresPlugin.h )

add_library( HARDIMeasuresPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( HARDIMeasuresPlugin vtkHARDI bmia_libs )

endif(PLUGINS_BUILD_HARDIMEASURES)